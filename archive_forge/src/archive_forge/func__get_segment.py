import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
def _get_segment(p0, p1, p2, p3, thickness=1, prev_miter=None, prev_scale=None):
    """Computes a line segment between the points p1 and p2.

    If points p0 or p3 are supplied then the segment p1->p2 will have the correct "miter" angle
    for each end respectively.  This returns computed miter and scale values which can be supplied
    to the next call of the method for a minor performance improvement.  If they are not supplied
    then they will be computed.

    :Parameters:
        `p0` : (float, float)
            The "previous" point for the segment p1->p2 which is used to compute the "miter"
            angle of the start of the segment.  If None is supplied then the start of the line
            is 90 degrees to the segment p1->p2.
        `p1` : (float, float)
            The origin of the segment p1->p2.
        `p2` : (float, float)
            The end of the segment p1->p2
        `p3` : (float, float)
            The "following" point for the segment p1->p2 which is used to compute the "miter"
            angle to the end of the segment.  If None is supplied then the end of the line is
            90 degrees to the segment p1->p2.
        `prev_miter`: pyglet.math.Vec2
            The miter value to be used.

    :type: (pyglet.math.Vec2, pyglet.math.Vec2, float, float, float, float, float, float)
    """
    v_np1p2 = Vec2(p2[0] - p1[0], p2[1] - p1[1]).normalize()
    v_normal = Vec2(-v_np1p2.y, v_np1p2.x)
    v_miter2 = v_normal
    scale1 = scale2 = thickness / 2.0
    v_miter1 = v_normal
    if prev_miter and prev_scale:
        v_miter1 = prev_miter
        scale1 = prev_scale
    elif p0:
        v_np0p1 = Vec2(p1[0] - p0[0], p1[1] - p0[1]).normalize()
        v_normal_p0p1 = Vec2(-v_np0p1.y, v_np0p1.x)
        v_miter1 = Vec2(v_normal_p0p1.x + v_normal.x, v_normal_p0p1.y + v_normal.y).normalize()
        scale1 = scale1 / math.sin(math.acos(v_np1p2.dot(v_miter1)))
    if p3:
        v_np2p3 = Vec2(p3[0] - p2[0], p3[1] - p2[1]).normalize()
        v_normal_p2p3 = Vec2(-v_np2p3.y, v_np2p3.x)
        v_miter2 = Vec2(v_normal_p2p3.x + v_normal.x, v_normal_p2p3.y + v_normal.y).normalize()
        scale2 = scale2 / math.sin(math.acos(v_np2p3.dot(v_miter2)))
    miter1_scaled_p = (v_miter1.x * scale1, v_miter1.y * scale1)
    miter2_scaled_p = (v_miter2.x * scale2, v_miter2.y * scale2)
    v1 = (p1[0] + miter1_scaled_p[0], p1[1] + miter1_scaled_p[1])
    v2 = (p2[0] + miter2_scaled_p[0], p2[1] + miter2_scaled_p[1])
    v3 = (p1[0] - miter1_scaled_p[0], p1[1] - miter1_scaled_p[1])
    v4 = (p2[0] + miter2_scaled_p[0], p2[1] + miter2_scaled_p[1])
    v5 = (p2[0] - miter2_scaled_p[0], p2[1] - miter2_scaled_p[1])
    v6 = (p1[0] - miter1_scaled_p[0], p1[1] - miter1_scaled_p[1])
    return (v_miter2, scale2, v1[0], v1[1], v2[0], v2[1], v3[0], v3[1], v4[0], v4[1], v5[0], v5[1], v6[0], v6[1])