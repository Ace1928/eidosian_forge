import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def _init_2021(self) -> None:
    """Initialize based on 2021 layout"""
    self.all_categories = sorted(os.listdir(self.root))
    self.categories_index = {k: {} for k in CATEGORIES_2021}
    for dir_index, dir_name in enumerate(self.all_categories):
        pieces = dir_name.split('_')
        if len(pieces) != 8:
            raise RuntimeError(f'Unexpected category name {dir_name}, wrong number of pieces')
        if pieces[0] != f'{dir_index:05d}':
            raise RuntimeError(f'Unexpected category id {pieces[0]}, expecting {dir_index:05d}')
        cat_map = {}
        for cat, name in zip(CATEGORIES_2021, pieces[1:7]):
            if name in self.categories_index[cat]:
                cat_id = self.categories_index[cat][name]
            else:
                cat_id = len(self.categories_index[cat])
                self.categories_index[cat][name] = cat_id
            cat_map[cat] = cat_id
        self.categories_map.append(cat_map)