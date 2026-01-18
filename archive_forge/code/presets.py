# Licensed under the GPL: https://www.gnu.org/licenses/gpl-3.0.en.html
# For details: reprotest/debian/copyright

import collections
import os
import re
import shlex
import subprocess

from reprotest.utils import AttributeFunctor


class ReprotestPreset(collections.namedtuple('_ReprotestPreset',
    'build_command artifact_pattern testbed_pre testbed_init testbed_build_pre source_pattern diffoscope_args')):
    """Named-tuple representing a reprotest command preset.

    You can manipulate it like this:

    >>> ReprotestPreset(None, None, None, None)
    ReprotestPreset(build_command=None, artifact_pattern=None, testbed_pre=None, testbed_init=None)

    >>> _.set.build_command("etc")
    ReprotestPreset(build_command='etc', artifact_pattern=None, testbed_pre=None, testbed_init=None)

    >>> _.append.build_command("; etc2")
    ReprotestPreset(build_command='etc; etc2', artifact_pattern=None, testbed_pre=None, testbed_init=None)

    >>> _.prepend.build_command("setup; ")
    ReprotestPreset(build_command='setup; etc; etc2', artifact_pattern=None, testbed_pre=None, testbed_init=None)

    >>> _.set.build_command("dpkg-buildpackage --no-sign -b")
    ReprotestPreset(build_command='dpkg-buildpackage --no-sign -b', artifact_pattern=None, testbed_pre=None, testbed_init=None)

    >>> _.str_replace.build_command(
    ...    "dpkg-buildpackage", "DEB_BUILD_OPTIONS=nocheck dpkg-buildpackage -Pnocheck")
    ReprotestPreset(build_command='DEB_BUILD_OPTIONS=nocheck dpkg-buildpackage -Pnocheck --no-sign -b', artifact_pattern=None, testbed_pre=None, testbed_init=None)
    """

    @property
    def set(self):
        """Set the given attribute to the given value."""
        return AttributeFunctor(self, lambda x, y: y)

    @property
    def str_replace(self):
        """Do a substring-replace on the given attribute."""
        return AttributeFunctor(self, str.replace)

    @property
    def prepend(self):
        """Prepend the given value to the given attribute."""
        return AttributeFunctor(self, lambda a, b: b + a)

    @property
    def append(self):
        """Apppend the given value to the given attribute."""
        return AttributeFunctor(self, lambda a, b: a + b)

    @property
    def re_replace(self):
        """Do a regex-replace on the given attribute."""
        return AttributeFunctor(self, lambda s, pattern, repl, **kwargs: re.sub(pattern, repl, s, **kwargs))


PRESET_DEB_DIR = ReprotestPreset(
    build_command='dpkg-buildpackage --no-sign -b',
    artifact_pattern='../*.*deb',
    testbed_pre=None,
    testbed_init=None,
    testbed_build_pre=None,
    source_pattern=None,
    diffoscope_args=[],
)

PRESET_RPM_DIR = ReprotestPreset(
    build_command='rpmbuild --rebuild',
    artifact_pattern='rpmbuild/RPMS/*.rpm',
    testbed_pre=None,
    testbed_init=None,
    testbed_build_pre=None,
    source_pattern='*.src.rpm',
    diffoscope_args=[],
)


# DEB
def preset_deb_schroot(fn, preset):
    return preset.set.testbed_init(
        # need to symlink /etc/mtab to work around a fusermount(1) deficiency
        'apt-get -y --no-install-recommends install disorderfs fakeroot faketime locales-all sudo util-linux; \
        test -c /dev/fuse || mknod -m 666 /dev/fuse c 10 229; \
        test -f /etc/mtab || ln -s ../proc/self/mounts /etc/mtab'
    ).set.testbed_build_pre(
        'apt-get -y --no-install-recommends build-dep ./"%s"' % fn)


def preset_deb_dsc(fn, aux):
    # we use "$(basename "$PWD")" so that the dirname varies iff we are varying the whole build path
    return PRESET_DEB_DIR.prepend.build_command(
            'dpkg-source -x "%s" "$(basename "$PWD")" && cd "$(basename "$PWD")" && ' % fn
        ).set.artifact_pattern("*.*deb"
        ).set.source_pattern(" ".join(shlex.quote(a) for a in [fn] + aux))


def parse_dsc_aux(path):
    dscfiles = subprocess.check_output(["egrep",
        # this regex comes from dcmd(1) from the devscripts package
        r"^ [0-9a-f]{32} [0-9]+ ((([a-zA-Z0-9_.-]+/)?[a-zA-Z0-9_.-]+|-) ([a-zA-Z]+|-) )?(.*)$",
        path])
    return [x.split()[-1].decode("utf-8") for x in dscfiles.splitlines()]


# RPM
def preset_rpm_rpmbuild(fn):
    # We ensure to have generated RPMs into temporary HOME folder and not in
    # rpmbuild/RPMS/%%{ARCH}/
    extra_build_command = [
        '--define "_build_name_fmt %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm"',
        '--define "_topdir $PWD/rpmbuild"',
        '%s' % fn
        ]

    # rpmbuild needs to be told to use reproducible specific options
    extra_build_command += ['--define "source_date_epoch_from_changelog Y"']
    extra_build_command += ['--define "clamp_mtime_to_source_date_epoch Y"']
    extra_build_command += ['--define "use_source_date_epoch_as_buildtime Y"']

    # Disable %clean stage: workaround issue when running with +fileordering
    extra_build_command += ['--noclean']
    extra_build_command += ['; rm -rf $PWD/rpmbuild/BUILD*']

    extra_build_command = ' '.join(extra_build_command)
    return PRESET_RPM_DIR.append.build_command(' %s' % extra_build_command)


###
def get_presets(buildfile, virtual_server):
    fn = os.path.basename(buildfile)
    parts = os.path.splitext(fn)
    if os.path.isdir(buildfile):
        if os.path.isdir(os.path.join(buildfile, "debian")):
            if virtual_server == "null":
                return PRESET_DEB_DIR
            else:
                return preset_deb_schroot(".", PRESET_DEB_DIR)
    elif os.path.isfile(buildfile):
        if parts[1] == '.dsc':
            if virtual_server == "null":
                return preset_deb_dsc(fn, parse_dsc_aux(buildfile))
            else:
                return preset_deb_schroot(fn, preset_deb_dsc(fn, parse_dsc_aux(buildfile)))
        elif parts[1] == '.rpm':
            if os.path.splitext(parts[0])[1] != '.src':
                raise ValueError('cannot determine .src.rpm')
            if virtual_server == "null":
                return preset_rpm_rpmbuild(fn)
            else:
                raise ValueError("only 'null' virtual_server is currently available for source RPM")
    raise ValueError('unrecognised file type: "%s"; try giving '
                     'an appropriate --build-command' % buildfile)
