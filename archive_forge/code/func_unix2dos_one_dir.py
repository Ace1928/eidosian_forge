import os
import re
import sys
def unix2dos_one_dir(modified_files, dir_name, file_names):
    for file in file_names:
        full_path = os.path.join(dir_name, file)
        unix2dos(full_path)
        if file is not None:
            modified_files.append(file)