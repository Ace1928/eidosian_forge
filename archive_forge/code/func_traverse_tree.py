import os
def traverse_tree(self, action, path, paths, checked):
    if path in checked:
        return
    self.traverse_path(action, path, paths, checked)
    if os.path.isdir(path):
        for fn in os.listdir(path):
            fn = os.path.join(path, fn)
            self.traverse_tree(action, fn, paths, checked)