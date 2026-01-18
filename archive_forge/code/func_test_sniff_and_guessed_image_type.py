import copy
from os.path import basename, dirname
from os.path import join as pjoin
import numpy as np
from .. import (
def test_sniff_and_guessed_image_type(img_klasses=all_image_classes):

    def test_image_class(img_path, expected_img_klass):
        """Compare an image of one image class to all others.

        The function should make sure that it loads the image with the expected
        class, but failing when given a bad sniff (when the sniff is used)."""

        def check_img(img_path, img_klass, sniff_mode, sniff, expect_success, msg):
            """Embedded function to do the actual checks expected."""
            if sniff_mode == 'no_sniff':
                is_img, new_sniff = img_klass.path_maybe_image(img_path)
            elif sniff_mode in ('empty', 'irrelevant', 'bad_sniff'):
                is_img, new_sniff = img_klass.path_maybe_image(img_path, (sniff, img_path))
            else:
                is_img, new_sniff = img_klass.path_maybe_image(img_path, sniff)
            if expect_success:
                new_msg = f'{img_klass.__name__} returned sniff==None ({msg})'
                expected_sizeof_hdr = getattr(img_klass.header_class, 'sizeof_hdr', 0)
                current_sizeof_hdr = 0 if new_sniff is None else len(new_sniff[0])
                assert current_sizeof_hdr >= expected_sizeof_hdr, new_msg
                new_msg = f'{basename(img_path)} ({msg}) image is{('' if is_img else ' not')} a {img_klass.__name__} image.'
                assert is_img, new_msg
            if sniff_mode == 'vanilla':
                return new_sniff
            else:
                return sniff
        sizeof_hdr = getattr(expected_img_klass.header_class, 'sizeof_hdr', 0)
        for sniff_mode, sniff in dict(vanilla=None, no_sniff=None, none=None, empty=b'', irrelevant=b'a' * (sizeof_hdr - 1), bad_sniff=b'a' * sizeof_hdr).items():
            for klass in img_klasses:
                if klass == expected_img_klass:
                    expect_success = sniff_mode != 'bad_sniff' or sizeof_hdr == 0
                else:
                    expect_success = False
                msg = f'{expected_img_klass.__name__}/ {sniff_mode}/ {expect_success}'
                sniff = check_img(img_path, klass, sniff_mode=sniff_mode, sniff=sniff, expect_success=expect_success, msg=msg)
    for img_filename, image_klass in [('example4d.nii.gz', Nifti1Image), ('nifti1.hdr', Nifti1Pair), ('example_nifti2.nii.gz', Nifti2Image), ('nifti2.hdr', Nifti2Pair), ('tiny.mnc', Minc1Image), ('small.mnc', Minc2Image), ('test.mgz', MGHImage), ('analyze.hdr', Spm2AnalyzeImage)]:
        test_image_class(pjoin(DATA_PATH, img_filename), image_klass)