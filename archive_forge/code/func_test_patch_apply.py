import os
import shutil
import tempfile
from io import BytesIO
from dulwich import porcelain
from ...repo import Repo
from .utils import CompatTestCase, run_git_or_fail
def test_patch_apply(self):
    file_list = ['to_exists', 'to_modify', 'to_delete']
    for file in file_list:
        file_path = os.path.join(self.repo_path, file)
        with open(file_path, 'w'):
            pass
    self.repo.stage(file_list)
    first_commit = self.repo.do_commit(b'The first commit')
    copy_path = os.path.join(self.test_dir, 'copy')
    shutil.copytree(self.repo_path, copy_path)
    with open(os.path.join(self.repo_path, 'to_modify'), 'w') as f:
        f.write('Modified!')
    os.remove(os.path.join(self.repo_path, 'to_delete'))
    with open(os.path.join(self.repo_path, 'to_add'), 'w'):
        pass
    self.repo.stage(['to_modify', 'to_delete', 'to_add'])
    second_commit = self.repo.do_commit(b'The second commit')
    first_tree = self.repo[first_commit].tree
    second_tree = self.repo[second_commit].tree
    outstream = BytesIO()
    porcelain.diff_tree(self.repo.path, first_tree, second_tree, outstream=outstream)
    patch_path = os.path.join(self.test_dir, 'patch.patch')
    with open(patch_path, 'wb') as patch:
        patch.write(outstream.getvalue())
    git_command = ['-C', copy_path, 'apply', patch_path]
    run_git_or_fail(git_command)
    original_files = set(os.listdir(self.repo_path))
    new_files = set(os.listdir(copy_path))
    self.assertEqual(original_files, new_files)
    for file in original_files:
        if file == '.git':
            continue
        original_file_path = os.path.join(self.repo_path, file)
        copy_file_path = os.path.join(copy_path, file)
        self.assertTrue(os.path.isfile(copy_file_path))
        with open(original_file_path, 'rb') as original_file:
            original_content = original_file.read()
        with open(copy_file_path, 'rb') as copy_file:
            copy_content = copy_file.read()
        self.assertEqual(original_content, copy_content)