import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
class LFWPeople(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold`` (default).
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root: str, split: str='10fold', image_set: str='funneled', transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, download: bool=False) -> None:
        super().__init__(root, split, image_set, 'people', transform, target_transform, download)
        self.class_to_idx = self._get_classes()
        self.data, self.targets = self._get_people()

    def _get_people(self) -> Tuple[List[str], List[int]]:
        data, targets = ([], [])
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()
            n_folds, s = (int(lines[0]), 1) if self.split == '10fold' else (1, 0)
            for fold in range(n_folds):
                n_lines = int(lines[s])
                people = [line.strip().split('\t') for line in lines[s + 1:s + n_lines + 1]]
                s += n_lines + 1
                for i, (identity, num_imgs) in enumerate(people):
                    for num in range(1, int(num_imgs) + 1):
                        img = self._get_path(identity, num)
                        data.append(img)
                        targets.append(self.class_to_idx[identity])
        return (data, targets)

    def _get_classes(self) -> Dict[str, int]:
        with open(os.path.join(self.root, self.names)) as f:
            lines = f.readlines()
            names = [line.strip().split()[0] for line in lines]
        class_to_idx = {name: i for i, name in enumerate(names)}
        return class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the identity of the person.
        """
        img = self._loader(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    def extra_repr(self) -> str:
        return super().extra_repr() + f'\nClasses (identities): {len(self.class_to_idx)}'