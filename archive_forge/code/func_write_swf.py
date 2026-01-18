import os
import zlib
import time  # noqa
import logging
import numpy as np
def write_swf(filename, images, duration=0.1, repeat=True):
    """Write an swf-file from the specified images. If repeat is False,
    the movie is finished with a stop action. Duration may also
    be a list with durations for each frame (note that the duration
    for each frame is always an integer amount of the minimum duration.)

    Images should be a list consisting numpy arrays with values between
    0 and 255 for integer types, and between 0 and 1 for float types.

    """
    images2 = checkImages(images)
    taglist = [FileAttributesTag(), SetBackgroundTag(0, 0, 0)]
    if hasattr(duration, '__len__'):
        if len(duration) == len(images2):
            duration = [d for d in duration]
        else:
            raise ValueError("len(duration) doesn't match amount of images.")
    else:
        duration = [duration for im in images2]
    minDuration = float(min(duration))
    delays = [round(d / minDuration) for d in duration]
    delays = [max(1, int(d)) for d in delays]
    fps = 1.0 / minDuration
    nframes = 0
    for im in images2:
        bm = BitmapTag(im)
        wh = (im.shape[1], im.shape[0])
        sh = ShapeTag(bm.id, (0, 0), wh)
        po = PlaceObjectTag(1, sh.id, move=nframes > 0)
        taglist.extend([bm, sh, po])
        for i in range(delays[nframes]):
            taglist.append(ShowFrameTag())
        nframes += 1
    if not repeat:
        taglist.append(DoActionTag('stop'))
    fp = open(filename, 'wb')
    try:
        build_file(fp, taglist, nframes=nframes, framesize=wh, fps=fps)
    except Exception:
        raise
    finally:
        fp.close()