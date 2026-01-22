import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
class _LFW(VisionDataset):
    base_folder = 'lfw-py'
    download_url_prefix = 'http://vis-www.cs.umass.edu/lfw/'
    file_dict = {'original': ('lfw', 'lfw.tgz', 'a17d05bd522c52d84eca14327a23d494'), 'funneled': ('lfw_funneled', 'lfw-funneled.tgz', '1b42dfed7d15c9b2dd63d5e5840c86ad'), 'deepfunneled': ('lfw-deepfunneled', 'lfw-deepfunneled.tgz', '68331da3eb755a505a502b5aacb3c201')}
    checksums = {'pairs.txt': '9f1ba174e4e1c508ff7cdf10ac338a7d', 'pairsDevTest.txt': '5132f7440eb68cf58910c8a45a2ac10b', 'pairsDevTrain.txt': '4f27cbf15b2da4a85c1907eb4181ad21', 'people.txt': '450f0863dd89e85e73936a6d71a3474b', 'peopleDevTest.txt': 'e4bf5be0a43b5dcd9dc5ccfcb8fb19c5', 'peopleDevTrain.txt': '54eaac34beb6d042ed3a7d883e247a21', 'lfw-names.txt': 'a6d0a479bd074669f656265a6e693f6d'}
    annot_file = {'10fold': '', 'train': 'DevTrain', 'test': 'DevTest'}
    names = 'lfw-names.txt'

    def __init__(self, root: str, split: str, image_set: str, view: str, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, download: bool=False) -> None:
        super().__init__(os.path.join(root, self.base_folder), transform=transform, target_transform=target_transform)
        self.image_set = verify_str_arg(image_set.lower(), 'image_set', self.file_dict.keys())
        images_dir, self.filename, self.md5 = self.file_dict[self.image_set]
        self.view = verify_str_arg(view.lower(), 'view', ['people', 'pairs'])
        self.split = verify_str_arg(split.lower(), 'split', ['10fold', 'train', 'test'])
        self.labels_file = f'{self.view}{self.annot_file[self.split]}.txt'
        self.data: List[Any] = []
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
        self.images_dir = os.path.join(self.root, images_dir)

    def _loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _check_integrity(self) -> bool:
        st1 = check_integrity(os.path.join(self.root, self.filename), self.md5)
        st2 = check_integrity(os.path.join(self.root, self.labels_file), self.checksums[self.labels_file])
        if not st1 or not st2:
            return False
        if self.view == 'people':
            return check_integrity(os.path.join(self.root, self.names), self.checksums[self.names])
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        url = f'{self.download_url_prefix}{self.filename}'
        download_and_extract_archive(url, self.root, filename=self.filename, md5=self.md5)
        download_url(f'{self.download_url_prefix}{self.labels_file}', self.root)
        if self.view == 'people':
            download_url(f'{self.download_url_prefix}{self.names}', self.root)

    def _get_path(self, identity: str, no: Union[int, str]) -> str:
        return os.path.join(self.images_dir, identity, f'{identity}_{int(no):04d}.jpg')

    def extra_repr(self) -> str:
        return f'Alignment: {self.image_set}\nSplit: {self.split}'

    def __len__(self) -> int:
        return len(self.data)