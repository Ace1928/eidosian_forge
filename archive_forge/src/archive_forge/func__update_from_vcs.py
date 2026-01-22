import os, subprocess, json
def _update_from_vcs(self, output):
    """Update state based on the VCS state e.g the output of git describe"""
    split = output[1:].split('-')
    if 'dev' in split[0]:
        dev_split = split[0].split('dev')
        self.dev = int(dev_split[1])
        split[0] = dev_split[0]
        if split[0].endswith('.'):
            split[0] = dev_split[0][:-1]
    self._release = tuple((int(el) for el in split[0].split('.')))
    self._commit_count = int(split[1])
    self._commit = str(split[2][1:])
    self._dirty = split[-1] == 'dirty'
    return self