from __future__ import annotations
def _make_version(major: int, minor: int, micro: int, releaselevel: str='final', serial: int=0, dev: int=0) -> str:
    """Create a readable version string from version_info tuple components."""
    assert releaselevel in ['alpha', 'beta', 'candidate', 'final']
    version = '%d.%d.%d' % (major, minor, micro)
    if releaselevel != 'final':
        short = {'alpha': 'a', 'beta': 'b', 'candidate': 'rc'}[releaselevel]
        version += f'{short}{serial}'
    if dev != 0:
        version += f'.dev{dev}'
    return version