import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _create_overlay(self):
    topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
    bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
    if self._render_every_frame:
        self.add_overlay(topleft, '', '')
    else:
        self.add_overlay(topleft, 'Run speed = %.3f x real time' % self._run_speed, '[S]lower, [F]aster')
    self.add_overlay(topleft, 'Ren[d]er every frame', 'On' if self._render_every_frame else 'Off')
    self.add_overlay(topleft, 'Switch camera (#cams = %d)' % (self.model.ncam + 1), '[Tab] (camera ID = %d)' % self.cam.fixedcamid)
    self.add_overlay(topleft, '[C]ontact forces', 'On' if self._contacts else 'Off')
    self.add_overlay(topleft, 'T[r]ansparent', 'On' if self._transparent else 'Off')
    if self._paused is not None:
        if not self._paused:
            self.add_overlay(topleft, 'Stop', '[Space]')
        else:
            self.add_overlay(topleft, 'Start', '[Space]')
            self.add_overlay(topleft, 'Advance simulation by one step', '[right arrow]')
    self.add_overlay(topleft, 'Referenc[e] frames', 'On' if self.vopt.frame == 1 else 'Off')
    self.add_overlay(topleft, '[H]ide Menu', '')
    if self._image_idx > 0:
        fname = self._image_path % (self._image_idx - 1)
        self.add_overlay(topleft, 'Cap[t]ure frame', 'Saved as %s' % fname)
    else:
        self.add_overlay(topleft, 'Cap[t]ure frame', '')
    self.add_overlay(topleft, 'Toggle geomgroup visibility', '0-4')
    self.add_overlay(bottomleft, 'FPS', '%d%s' % (1 / self._time_per_render, ''))
    self.add_overlay(bottomleft, 'Solver iterations', str(self.data.solver_iter + 1))
    self.add_overlay(bottomleft, 'Step', str(round(self.data.time / self.model.opt.timestep)))
    self.add_overlay(bottomleft, 'timestep', '%.5f' % self.model.opt.timestep)