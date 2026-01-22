from __future__ import division, absolute_import
import sys
import os
import datetime
from twisted.python.filepath import FilePath
from twisted.python.compat import NativeStringIO
from twisted.trial.unittest import TestCase
from incremental.update import _run, run
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
class CreatedUpdateTests(TestCase):
    maxDiff = None

    def setUp(self):
        self.srcdir = FilePath(self.mktemp())
        self.srcdir.makedirs()
        packagedir = self.srcdir.child('inctestpkg')
        packagedir.makedirs()
        packagedir.child('__init__.py').setContent(b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", "NEXT", 0, 0).short()\nnext_released_version = "inctestpkg NEXT"\n')
        packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3)\n__all__ = ["__version__"]\n')
        self.getcwd = lambda: self.srcdir.path
        self.packagedir = packagedir

        class Date(object):
            year = 2016
            month = 8
        self.date = Date()

    def test_path(self):
        """
        `incremental.update package --path=<path> --dev` increments the dev
        version of the package on the given path
        """
        out = []
        _run(u'inctestpkg', path=self.packagedir.path, newversion=None, patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _print=out.append)
        self.assertTrue(self.packagedir.child('_version.py').exists())
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, dev=0)\n__all__ = ["__version__"]\n')

    def test_dev(self):
        """
        `incremental.update package --dev` increments the dev version.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertTrue(self.packagedir.child('_version.py').exists())
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, dev=0)\n__all__ = ["__version__"]\n')

    def test_patch(self):
        """
        `incremental.update package --patch` increments the patch version.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 4)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 2, 4).short()\nnext_released_version = "inctestpkg 1.2.4"\n')

    def test_patch_with_prerelease_and_dev(self):
        """
        `incremental.update package --patch` increments the patch version, and
        disregards any old prerelease/dev versions.
        """
        self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, release_candidate=1, dev=2)\n__all__ = ["__version__"]\n')
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 4)\n__all__ = ["__version__"]\n')

    def test_rc_patch(self):
        """
        `incremental.update package --patch --rc` increments the patch
        version and makes it a release candidate.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 4, release_candidate=1)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 2, 4, release_candidate=1).short()\nnext_released_version = "inctestpkg 1.2.4.rc1"\n')

    def test_rc_with_existing_rc(self):
        """
        `incremental.update package --rc` increments the rc version if the
        existing version is an rc, and discards any dev version.
        """
        self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, release_candidate=1, dev=2)\n__all__ = ["__version__"]\n')
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, release_candidate=2)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 2, 3, release_candidate=2).short()\nnext_released_version = "inctestpkg 1.2.3.rc2"\n')

    def test_rc_with_no_rc(self):
        """
        `incremental.update package --rc`, when the package is not a release
        candidate, will issue a new major/minor rc, and disregards the micro
        and dev.
        """
        self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, dev=2)\n__all__ = ["__version__"]\n')
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 16, 8, 0, release_candidate=1)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 16, 8, 0, release_candidate=1).short()\nnext_released_version = "inctestpkg 16.8.0.rc1"\n')

    def test_full_with_rc(self):
        """
        `incremental.update package`, when the package is a release
        candidate, will issue the major/minor, sans release candidate or dev.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 16, 8, 0, release_candidate=1)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 16, 8, 0, release_candidate=1).short()\nnext_released_version = "inctestpkg 16.8.0.rc1"\n')
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 16, 8, 0)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 16, 8, 0).short()\nnext_released_version = "inctestpkg 16.8.0"\n')

    def test_full_without_rc(self):
        """
        `incremental.update package`, when the package is NOT a release
        candidate, will raise an error.
        """
        out = []
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'You need to issue a rc before updating the major/minor')

    def test_post(self):
        """
        `incremental.update package --post` increments the post version.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=True, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertTrue(self.packagedir.child('_version.py').exists())
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, post=0)\n__all__ = ["__version__"]\n')

    def test_post_with_prerelease_and_dev(self):
        """
        `incremental.update package --post` increments the post version, and
        disregards any old prerelease/dev versions.
        """
        self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, release_candidate=1, dev=2)\n__all__ = ["__version__"]\n')
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=True, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, post=0)\n__all__ = ["__version__"]\n')

    def test_post_with_existing_post(self):
        """
        `incremental.update package --post` increments the post version if the
        existing version is an postrelease, and discards any dev version.
        """
        self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, post=1, dev=2)\n__all__ = ["__version__"]\n')
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=True, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, post=2)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 2, 3, post=2).short()\nnext_released_version = "inctestpkg 1.2.3.post2"\n')

    def test_no_mix_newversion(self):
        """
        The `--newversion` flag can't be mixed with --patch, --rc, --post,
        or --dev.
        """
        out = []
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion='1', patch=True, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --newversion')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --newversion')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=False, post=True, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --newversion')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --newversion')

    def test_no_mix_dev(self):
        """
        The `--dev` flag can't be mixed with --patch, --rc, or --post.
        """
        out = []
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --dev')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=True, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --dev')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=True, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --dev')

    def test_no_mix_create(self):
        """
        The `--create` flag can't be mixed with --patch, --rc, --post,
        --dev, or --newversion.
        """
        out = []
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=False, post=False, dev=False, create=True, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --create')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=False, post=False, dev=False, create=True, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --create')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=True, post=False, dev=False, create=True, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --create')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=True, dev=False, create=True, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --create')
        with self.assertRaises(ValueError) as e:
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=True, create=True, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(e.exception.args[0], 'Only give --create')

    def test_newversion(self):
        """
        `incremental.update package --newversion=1.2.3.rc1.post2.dev3`, will
        set that version in the package.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion='1.2.3.rc1.post2.dev3', patch=False, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, release_candidate=1, post=2, dev=3)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 2, 3, release_candidate=1, post=2, dev=3).short()\nnext_released_version = "inctestpkg 1.2.3.rc1.post2.dev3"\n')

    def test_newversion_bare(self):
        """
        `incremental.update package --newversion=1`, will set that
        version in the package.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion='1', patch=False, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 0, 0)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 0, 0).short()\nnext_released_version = "inctestpkg 1.0.0"\n')

    def test_newversion_bare_major_minor(self):
        """
        `incremental.update package --newversion=1.1`, will set that
        version in the package.
        """
        out = []
        _run(u'inctestpkg', path=None, newversion='1.1', patch=False, rc=False, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 1, 0)\n__all__ = ["__version__"]\n')
        self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 1, 0).short()\nnext_released_version = "inctestpkg 1.1.0"\n')