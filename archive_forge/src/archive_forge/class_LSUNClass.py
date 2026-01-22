import io
import os.path
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from PIL import Image
from .utils import iterable_to_str, verify_str_arg
from .vision import VisionDataset
class LSUNClass(VisionDataset):

    def __init__(self, root: str, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None) -> None:
        import lmdb
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join((c for c in root if c in string.ascii_letters))
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, 'rb'))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, 'wb'))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = (None, None)
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    def __len__(self) -> int:
        return self.length