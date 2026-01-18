import os
import os.path
import time
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect

Native support for Multitouch devices on Linux, using libmtdev.
===============================================================

The Mtdev project is a part of the Ubuntu Maverick multitouch architecture.
You can read more on http://wiki.ubuntu.com/Multitouch

To configure MTDev, it's preferable to use probesysfs providers.
Check :py:class:`~kivy.input.providers.probesysfs` for more information.

Otherwise, add this to your configuration::

    [input]
    # devicename = hidinput,/dev/input/eventXX
    acert230h = mtdev,/dev/input/event2

.. note::
    You must have read access to the input event.

You can use a custom range for the X, Y and pressure values.
On some drivers, the range reported is invalid.
To fix that, you can add these options to the argument line:

* invert_x : 1 to invert X axis
* invert_y : 1 to invert Y axis
* min_position_x : X minimum
* max_position_x : X maximum
* min_position_y : Y minimum
* max_position_y : Y maximum
* min_pressure : pressure minimum
* max_pressure : pressure maximum
* min_touch_major : width shape minimum
* max_touch_major : width shape maximum
* min_touch_minor : width shape minimum
* max_touch_minor : height shape maximum
* rotation : 0,90,180 or 270 to rotate

An inverted display configuration will look like this::

    [input]
    # example for inverting touch events
    display = mtdev,/dev/input/event0,invert_x=1,invert_y=1
