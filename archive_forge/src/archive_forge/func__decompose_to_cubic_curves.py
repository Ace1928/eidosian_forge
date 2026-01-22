from fontTools.misc.transform import Identity, Scale
from math import atan2, ceil, cos, fabs, isfinite, pi, radians, sin, sqrt, tan
def _decompose_to_cubic_curves(self):
    if self.center_point is None and (not self._parametrize()):
        return
    point_transform = Identity.rotate(self.angle).scale(self.rx, self.ry)
    num_segments = int(ceil(fabs(self.theta_arc / (PI_OVER_TWO + 0.001))))
    for i in range(num_segments):
        start_theta = self.theta1 + i * self.theta_arc / num_segments
        end_theta = self.theta1 + (i + 1) * self.theta_arc / num_segments
        t = 4 / 3 * tan(0.25 * (end_theta - start_theta))
        if not isfinite(t):
            return
        sin_start_theta = sin(start_theta)
        cos_start_theta = cos(start_theta)
        sin_end_theta = sin(end_theta)
        cos_end_theta = cos(end_theta)
        point1 = complex(cos_start_theta - t * sin_start_theta, sin_start_theta + t * cos_start_theta)
        point1 += self.center_point
        target_point = complex(cos_end_theta, sin_end_theta)
        target_point += self.center_point
        point2 = target_point
        point2 += complex(t * sin_end_theta, -t * cos_end_theta)
        point1 = _map_point(point_transform, point1)
        point2 = _map_point(point_transform, point2)
        target_point = _map_point(point_transform, target_point)
        yield (point1, point2, target_point)