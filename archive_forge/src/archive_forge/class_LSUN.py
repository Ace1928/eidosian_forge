import io
import os.path
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from PIL import Image
from .utils import iterable_to_str, verify_str_arg
from .vision import VisionDataset
class LSUN(VisionDataset):
    """`LSUN <https://www.yf.io/p/lsun>`_ dataset.

    You will need to install the ``lmdb`` package to use this dataset: run
    ``pip install lmdb``

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root: str, classes: Union[str, List[str]]='train', transform: Optional[Callable]=None, target_transform: Optional[Callable]=None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes = self._verify_classes(classes)
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(root=os.path.join(root, f'{c}_lmdb'), transform=transform))
        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)
        self.length = count

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']
        try:
            classes = cast(str, classes)
            verify_str_arg(classes, 'classes', dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = 'Expected type str or Iterable for argument classes, but got type {}.'
                raise ValueError(msg.format(type(classes)))
            classes = list(classes)
            msg_fmtstr_type = 'Expected type str for elements in argument classes, but got type {}.'
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = ('_'.join(c_short[:-1]), c_short[-1])
                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, 'LSUN class', iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)
                msg = msg_fmtstr.format(dset_opt, 'postfix', iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)
        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind
        db = self.dbs[target]
        index = index - sub
        if self.target_transform is not None:
            target = self.target_transform(target)
        img, _ = db[index]
        return (img, target)

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return 'Classes: {classes}'.format(**self.__dict__)