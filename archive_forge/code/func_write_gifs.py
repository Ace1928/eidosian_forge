from collections import defaultdict
import numpy as np
from moviepy.decorators import use_clip_fps_by_default
def write_gifs(self, clip, gif_dir):
    for start, end, _, _ in self:
        name = '%s/%08d_%08d.gif' % (gif_dir, 100 * start, 100 * end)
        clip.subclip(start, end).write_gif(name, verbose=False)