import numpy as np
 Returns a new clip with a mask for transparency where the original
    clip is of the given color.

    You can also have a "progressive" mask by specifying a non-nul distance
    threshold thr. In this case, if the distance between a pixel and the given
    color is d, the transparency will be 

    d**s / (thr**s + d**s)

    which is 1 when d>>thr and 0 for d<<thr, the stiffness of the effect being
    parametrized by s
    