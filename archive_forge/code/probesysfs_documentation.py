import os
from os.path import sep

Auto Create Input Provider Config Entry for Available MT Hardware (linux only).
===============================================================================

Thanks to Marc Tardif for the probing code, taken from scan-for-mt-device.

The device discovery is done by this provider. However, the reading of
input can be performed by other providers like: hidinput, mtdev and
linuxwacom. mtdev is used prior to other providers. For more
information about mtdev, check :py:class:`~kivy.input.providers.mtdev`.

Here is an example of auto creation::

    [input]
    # using mtdev
    device_%(name)s = probesysfs,provider=mtdev
    # using hidinput
    device_%(name)s = probesysfs,provider=hidinput
    # using mtdev with a match on name
    device_%(name)s = probesysfs,provider=mtdev,match=acer

    # using hidinput with custom parameters to hidinput (all on one line)
    %(name)s = probesysfs,
        provider=hidinput,param=min_pressure=1,param=max_pressure=99

    # you can also match your wacom touchscreen
    touch = probesysfs,match=E3 Finger,provider=linuxwacom,
        select_all=1,param=mode=touch
    # and your wacom pen
    pen = probesysfs,match=E3 Pen,provider=linuxwacom,
        select_all=1,param=mode=pen

By default, ProbeSysfs module will enumerate hardware from the /sys/class/input
device, and configure hardware with ABS_MT_POSITION_X capability. But for
example, the wacom screen doesn't support this capability. You can prevent this
behavior by putting select_all=1 in your config line. Add use_mouse=1 to also
include touchscreen hardware that offers core pointer functionality.
