import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
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