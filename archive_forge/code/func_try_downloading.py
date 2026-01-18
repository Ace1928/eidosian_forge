import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def try_downloading(directory, row):
    document_id, kind, story_url = (row['document_id'], row['kind'], row['story_url'])
    story_path = os.path.join(directory, document_id + '.content')
    actual_story_size = 0
    if os.path.exists(story_path):
        with open(story_path, 'rb') as f:
            actual_story_size = len(f.read())
    if actual_story_size <= 19000:
        if kind == 'gutenberg':
            time.sleep(2)
        build_data.download(story_url, directory, document_id + '.content')
    else:
        return True
    file_type = subprocess.check_output(['file', '-b', story_path])
    file_type = file_type.decode('utf-8')
    if 'gzip compressed' in file_type:
        gz_path = os.path.join(directory, document_id + '.content.gz')
        shutil.move(story_path, gz_path)
        build_data.untar(gz_path)
    return False