import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
class _Streamline(object):
    """
    Refer to FigureFactory.create_streamline() for docstring
    """

    def __init__(self, x, y, u, v, density, angle, arrow_scale, **kwargs):
        self.x = np.array(x)
        self.y = np.array(y)
        self.u = np.array(u)
        self.v = np.array(v)
        self.angle = angle
        self.arrow_scale = arrow_scale
        self.density = int(30 * density)
        self.delta_x = self.x[1] - self.x[0]
        self.delta_y = self.y[1] - self.y[0]
        self.val_x = self.x
        self.val_y = self.y
        self.blank = np.zeros((self.density, self.density))
        self.spacing_x = len(self.x) / float(self.density - 1)
        self.spacing_y = len(self.y) / float(self.density - 1)
        self.trajectories = []
        self.u = self.u / (self.x[-1] - self.x[0])
        self.v = self.v / (self.y[-1] - self.y[0])
        self.speed = np.sqrt(self.u ** 2 + self.v ** 2)
        self.u *= len(self.x)
        self.v *= len(self.y)
        self.st_x = []
        self.st_y = []
        self.get_streamlines()
        streamline_x, streamline_y = self.sum_streamlines()
        arrows_x, arrows_y = self.get_streamline_arrows()

    def blank_pos(self, xi, yi):
        """
        Set up positions for trajectories to be used with rk4 function.
        """
        return (int(xi / self.spacing_x + 0.5), int(yi / self.spacing_y + 0.5))

    def value_at(self, a, xi, yi):
        """
        Set up for RK4 function, based on Bokeh's streamline code
        """
        if isinstance(xi, np.ndarray):
            self.x = xi.astype(int)
            self.y = yi.astype(int)
        else:
            self.val_x = int(xi)
            self.val_y = int(yi)
        a00 = a[self.val_y, self.val_x]
        a01 = a[self.val_y, self.val_x + 1]
        a10 = a[self.val_y + 1, self.val_x]
        a11 = a[self.val_y + 1, self.val_x + 1]
        xt = xi - self.val_x
        yt = yi - self.val_y
        a0 = a00 * (1 - xt) + a01 * xt
        a1 = a10 * (1 - xt) + a11 * xt
        return a0 * (1 - yt) + a1 * yt

    def rk4_integrate(self, x0, y0):
        """
        RK4 forward and back trajectories from the initial conditions.

        Adapted from Bokeh's streamline -uses Runge-Kutta method to fill
        x and y trajectories then checks length of traj (s in units of axes)
        """

        def f(xi, yi):
            dt_ds = 1.0 / self.value_at(self.speed, xi, yi)
            ui = self.value_at(self.u, xi, yi)
            vi = self.value_at(self.v, xi, yi)
            return (ui * dt_ds, vi * dt_ds)

        def g(xi, yi):
            dt_ds = 1.0 / self.value_at(self.speed, xi, yi)
            ui = self.value_at(self.u, xi, yi)
            vi = self.value_at(self.v, xi, yi)
            return (-ui * dt_ds, -vi * dt_ds)
        check = lambda xi, yi: 0 <= xi < len(self.x) - 1 and 0 <= yi < len(self.y) - 1
        xb_changes = []
        yb_changes = []

        def rk4(x0, y0, f):
            ds = 0.01
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = self.blank_pos(xi, yi)
            xf_traj = []
            yf_traj = []
            while check(xi, yi):
                xf_traj.append(xi)
                yf_traj.append(yi)
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + 0.5 * ds * k1x, yi + 0.5 * ds * k1y)
                    k3x, k3y = f(xi + 0.5 * ds * k2x, yi + 0.5 * ds * k2y)
                    k4x, k4y = f(xi + ds * k3x, yi + ds * k3y)
                except IndexError:
                    break
                xi += ds * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
                yi += ds * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
                if not check(xi, yi):
                    break
                stotal += ds
                new_xb, new_yb = self.blank_pos(xi, yi)
                if new_xb != xb or new_yb != yb:
                    if self.blank[new_yb, new_xb] == 0:
                        self.blank[new_yb, new_xb] = 1
                        xb_changes.append(new_xb)
                        yb_changes.append(new_yb)
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return (stotal, xf_traj, yf_traj)
        sf, xf_traj, yf_traj = rk4(x0, y0, f)
        sb, xb_traj, yb_traj = rk4(x0, y0, g)
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]
        if len(x_traj) < 1:
            return None
        if stotal > 0.2:
            initxb, inityb = self.blank_pos(x0, y0)
            self.blank[inityb, initxb] = 1
            return (x_traj, y_traj)
        else:
            for xb, yb in zip(xb_changes, yb_changes):
                self.blank[yb, xb] = 0
            return None

    def traj(self, xb, yb):
        """
        Integrate trajectories

        :param (int) xb: results of passing xi through self.blank_pos
        :param (int) xy: results of passing yi through self.blank_pos

        Calculate each trajectory based on rk4 integrate method.
        """
        if xb < 0 or xb >= self.density or yb < 0 or (yb >= self.density):
            return
        if self.blank[yb, xb] == 0:
            t = self.rk4_integrate(xb * self.spacing_x, yb * self.spacing_y)
            if t is not None:
                self.trajectories.append(t)

    def get_streamlines(self):
        """
        Get streamlines by building trajectory set.
        """
        for indent in range(self.density // 2):
            for xi in range(self.density - 2 * indent):
                self.traj(xi + indent, indent)
                self.traj(xi + indent, self.density - 1 - indent)
                self.traj(indent, xi + indent)
                self.traj(self.density - 1 - indent, xi + indent)
        self.st_x = [np.array(t[0]) * self.delta_x + self.x[0] for t in self.trajectories]
        self.st_y = [np.array(t[1]) * self.delta_y + self.y[0] for t in self.trajectories]
        for index in range(len(self.st_x)):
            self.st_x[index] = self.st_x[index].tolist()
            self.st_x[index].append(np.nan)
        for index in range(len(self.st_y)):
            self.st_y[index] = self.st_y[index].tolist()
            self.st_y[index].append(np.nan)

    def get_streamline_arrows(self):
        """
        Makes an arrow for each streamline.

        Gets angle of streamline at 1/3 mark and creates arrow coordinates
        based off of user defined angle and arrow_scale.

        :param (array) st_x: x-values for all streamlines
        :param (array) st_y: y-values for all streamlines
        :param (angle in radians) angle: angle of arrowhead. Default = pi/9
        :param (float in [0,1]) arrow_scale: value to scale length of arrowhead
            Default = .09
        :rtype (list, list) arrows_x: x-values to create arrowhead and
            arrows_y: y-values to create arrowhead
        """
        arrow_end_x = np.empty(len(self.st_x))
        arrow_end_y = np.empty(len(self.st_y))
        arrow_start_x = np.empty(len(self.st_x))
        arrow_start_y = np.empty(len(self.st_y))
        for index in range(len(self.st_x)):
            arrow_end_x[index] = self.st_x[index][int(len(self.st_x[index]) / 3)]
            arrow_start_x[index] = self.st_x[index][int(len(self.st_x[index]) / 3) - 1]
            arrow_end_y[index] = self.st_y[index][int(len(self.st_y[index]) / 3)]
            arrow_start_y[index] = self.st_y[index][int(len(self.st_y[index]) / 3) - 1]
        dif_x = arrow_end_x - arrow_start_x
        dif_y = arrow_end_y - arrow_start_y
        orig_err = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')
        streamline_ang = np.arctan(dif_y / dif_x)
        np.seterr(**orig_err)
        ang1 = streamline_ang + self.angle
        ang2 = streamline_ang - self.angle
        seg1_x = np.cos(ang1) * self.arrow_scale
        seg1_y = np.sin(ang1) * self.arrow_scale
        seg2_x = np.cos(ang2) * self.arrow_scale
        seg2_y = np.sin(ang2) * self.arrow_scale
        point1_x = np.empty(len(dif_x))
        point1_y = np.empty(len(dif_y))
        point2_x = np.empty(len(dif_x))
        point2_y = np.empty(len(dif_y))
        for index in range(len(dif_x)):
            if dif_x[index] >= 0:
                point1_x[index] = arrow_end_x[index] - seg1_x[index]
                point1_y[index] = arrow_end_y[index] - seg1_y[index]
                point2_x[index] = arrow_end_x[index] - seg2_x[index]
                point2_y[index] = arrow_end_y[index] - seg2_y[index]
            else:
                point1_x[index] = arrow_end_x[index] + seg1_x[index]
                point1_y[index] = arrow_end_y[index] + seg1_y[index]
                point2_x[index] = arrow_end_x[index] + seg2_x[index]
                point2_y[index] = arrow_end_y[index] + seg2_y[index]
        space = np.empty(len(point1_x))
        space[:] = np.nan
        arrows_x = np.array([point1_x, arrow_end_x, point2_x, space])
        arrows_x = arrows_x.flatten('F')
        arrows_x = arrows_x.tolist()
        arrows_y = np.array([point1_y, arrow_end_y, point2_y, space])
        arrows_y = arrows_y.flatten('F')
        arrows_y = arrows_y.tolist()
        return (arrows_x, arrows_y)

    def sum_streamlines(self):
        """
        Makes all streamlines readable as a single trace.

        :rtype (list, list): streamline_x: all x values for each streamline
            combined into single list and streamline_y: all y values for each
            streamline combined into single list
        """
        streamline_x = sum(self.st_x, [])
        streamline_y = sum(self.st_y, [])
        return (streamline_x, streamline_y)