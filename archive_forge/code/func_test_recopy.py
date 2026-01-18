import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_recopy(_temp_analyze_files):
    orig_img, orig_hdr = _temp_analyze_files
    pth, fname = os.path.split(orig_img)
    img_link = os.path.join(pth, 'imglink.img')
    new_img = os.path.join(pth, 'newfile.img')
    new_hdr = os.path.join(pth, 'newfile.hdr')
    copyfile(orig_img, img_link)
    for copy in (True, False):
        for use_hardlink in (True, False):
            for hashmethod in ('timestamp', 'content'):
                kwargs = {'copy': copy, 'use_hardlink': use_hardlink, 'hashmethod': hashmethod}
                if copy and (not use_hardlink) and (hashmethod == 'timestamp'):
                    continue
                copyfile(orig_img, new_img, **kwargs)
                img_stat = _ignore_atime(os.stat(new_img))
                hdr_stat = _ignore_atime(os.stat(new_hdr))
                copyfile(orig_img, new_img, **kwargs)
                err_msg = 'Regular - OS: {}; Copy: {}; Hardlink: {}'.format(os.name, copy, use_hardlink)
                assert img_stat == _ignore_atime(os.stat(new_img)), err_msg
                assert hdr_stat == _ignore_atime(os.stat(new_hdr)), err_msg
                os.unlink(new_img)
                os.unlink(new_hdr)
                copyfile(img_link, new_img, **kwargs)
                img_stat = _ignore_atime(os.stat(new_img))
                hdr_stat = _ignore_atime(os.stat(new_hdr))
                copyfile(img_link, new_img, **kwargs)
                err_msg = 'Symlink - OS: {}; Copy: {}; Hardlink: {}'.format(os.name, copy, use_hardlink)
                assert img_stat == _ignore_atime(os.stat(new_img)), err_msg
                assert hdr_stat == _ignore_atime(os.stat(new_hdr)), err_msg
                os.unlink(new_img)
                os.unlink(new_hdr)