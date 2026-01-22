import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
class DockerPlatformTests(SynchronousTestCase):
    """
    Tests for L{twisted.python.runtime.Platform.isDocker}.
    """

    def test_noChecksOnLinux(self) -> None:
        """
        If the platform is not Linux, C{isDocker()} always returns L{False}.
        """
        platform = Platform(None, 'win32')
        self.assertFalse(platform.isDocker())

    def test_noCGroups(self) -> None:
        """
        If the platform is Linux, and the cgroups file in C{/proc} does not
        exist, C{isDocker()} returns L{False}
        """
        platform = Platform(None, 'linux')
        self.assertFalse(platform.isDocker(_initCGroupLocation='fakepath'))

    def test_cgroupsSuggestsDocker(self) -> None:
        """
        If the platform is Linux, and the cgroups file (faked out here) exists,
        and one of the paths starts with C{/docker/}, C{isDocker()} returns
        C{True}.
        """
        cgroupsFile = self.mktemp()
        with open(cgroupsFile, 'wb') as f:
            f.write(b'10:debug:/\n9:net_prio:/\n8:perf_event:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f\n7:net_cls:/\n6:freezer:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f\n5:devices:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f\n4:blkio:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f\n3:cpuacct:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f\n2:cpu:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f\n1:cpuset:/docker/104155a6453cb67590027e397dc90fc25a06a7508403c797bc89ea43adf8d35f')
        platform = Platform(None, 'linux')
        self.assertTrue(platform.isDocker(_initCGroupLocation=cgroupsFile))

    def test_cgroupsSuggestsRealSystem(self) -> None:
        """
        If the platform is Linux, and the cgroups file (faked out here) exists,
        and none of the paths starts with C{/docker/}, C{isDocker()} returns
        C{False}.
        """
        cgroupsFile = self.mktemp()
        with open(cgroupsFile, 'wb') as f:
            f.write(b'9:perf_event:/\n8:blkio:/\n7:net_cls:/\n6:freezer:/\n5:devices:/\n4:memory:/\n3:cpuacct,cpu:/\n2:cpuset:/\n1:name=systemd:/system')
        platform = Platform(None, 'linux')
        self.assertFalse(platform.isDocker(_initCGroupLocation=cgroupsFile))