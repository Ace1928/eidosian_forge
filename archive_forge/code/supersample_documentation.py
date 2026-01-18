import numpy as np
 Replaces each frame at time t by the mean of `nframes` equally spaced frames
    taken in the interval [t-d, t+d]. This results in motion blur.