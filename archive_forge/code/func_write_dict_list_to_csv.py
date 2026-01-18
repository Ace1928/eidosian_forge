import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def write_dict_list_to_csv(dict_list, filepath):
    keys = list(dict_list[0].keys())
    with open(filepath, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in dict_list:
            writer.writerow(row)