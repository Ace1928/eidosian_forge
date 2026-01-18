import unittest
import io
import os
import tempfile
from kivy import setupconfig
@unittest.skip("Travis on Xenial don't have SDL_image >= 2.0.5")
def test_save_into_bytesio(self):
    Image = self.cls
    if setupconfig.PLATFORM == 'darwin':
        return
    img = Image.load('data/logo/kivy-icon-512.png')
    self.assertIsNotNone(img)
    with self.assertRaises(Exception) as context:
        bio = io.BytesIO()
        img.save(bio)
    bio = io.BytesIO()
    self.assertTrue(img.save(bio, fmt='png'))
    pngdata = bio.read()
    self.assertTrue(len(pngdata) > 0)
    try:
        _, filename = tempfile.mkstemp(suffix='.png')
        self.assertTrue(img.save(filename, fmt='png'))
    finally:
        os.unlink(filename)
    bio = io.BytesIO()
    self.assertTrue(img.save(bio, fmt='jpg'))
    self.assertTrue(len(bio.read()) > 0)
    with tempfile.NamedTemporaryFile(suffix='.jpg') as fd:
        self.assertTrue(img.save(fd.name))