from typing import Optional
import numpy as np
from numpy import cos, pi, sin
from gym import core, logger, spaces
from gym.error import DependencyNotInstalled
from gym.envs.classic_control import utils
def _dsdt(self, s_augmented):
    m1 = self.LINK_MASS_1
    m2 = self.LINK_MASS_2
    l1 = self.LINK_LENGTH_1
    lc1 = self.LINK_COM_POS_1
    lc2 = self.LINK_COM_POS_2
    I1 = self.LINK_MOI
    I2 = self.LINK_MOI
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
    phi1 = -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2) + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
    if self.book_or_nips == 'nips':
        ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    else:
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0)