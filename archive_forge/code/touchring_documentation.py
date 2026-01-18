from kivy.core.image import Image
from kivy.graphics import Color, Rectangle
from kivy import kivy_data_dir
from os.path import join

Touchring
=========

Shows rings around every touch on the surface / screen. You can use this module
to check that you don't have any calibration issues with touches.

Configuration
-------------

:Parameters:
    `image`: str, defaults to '<kivy>/data/images/ring.png'
        Filename of the image to use.
    `scale`: float, defaults to 1.
        Scale of the image.
    `alpha`: float, defaults to 1.
        Opacity of the image.

Example
-------

In your configuration (`~/.kivy/config.ini`), you can add something like
this::

    [modules]
    touchring = image=mypointer.png,scale=.3,alpha=.7

