import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
class SvgPath(Path, UserNode):
    """Path, from an svg path string"""

    def __init__(self, s, isClipPath=0, autoclose=None, fillMode=FILL_EVEN_ODD, **kw):
        vswap = kw.pop('vswap', 0)
        hswap = kw.pop('hswap', 0)
        super().__init__(points=None, operators=None, isClipPath=isClipPath, autoclose=autoclose, fillMode=fillMode, **kw)
        if not s:
            return
        normPath = normalise_svg_path(s)
        points = self.points
        unclosed_subpath_pointers = []
        subpath_start = []
        lastop = ''
        last_quadratic_cp = None
        for i in range(0, len(normPath), 2):
            op, nums = normPath[i:i + 2]
            if op in ('m', 'M') and i > 0 and (self.operators[-1] != _CLOSEPATH):
                unclosed_subpath_pointers.append(len(self.operators))
            if op == 'M':
                self.moveTo(*nums)
                subpath_start = points[-2:]
            elif op == 'L':
                self.lineTo(*nums)
            elif op == 'm':
                if len(points) >= 2:
                    if lastop in ('Z', 'z'):
                        starting_point = subpath_start
                    else:
                        starting_point = points[-2:]
                    xn, yn = (starting_point[0] + nums[0], starting_point[1] + nums[1])
                    self.moveTo(xn, yn)
                else:
                    self.moveTo(*nums)
                subpath_start = points[-2:]
            elif op == 'l':
                xn, yn = (points[-2] + nums[0], points[-1] + nums[1])
                self.lineTo(xn, yn)
            elif op == 'H':
                self.lineTo(nums[0], points[-1])
            elif op == 'V':
                self.lineTo(points[-2], nums[0])
            elif op == 'h':
                self.lineTo(points[-2] + nums[0], points[-1])
            elif op == 'v':
                self.lineTo(points[-2], points[-1] + nums[0])
            elif op == 'C':
                self.curveTo(*nums)
            elif op == 'S':
                x2, y2, xn, yn = nums
                if len(points) < 4 or lastop not in {'c', 'C', 's', 'S'}:
                    xp, yp, x0, y0 = points[-2:] * 2
                else:
                    xp, yp, x0, y0 = points[-4:]
                xi, yi = (x0 + (x0 - xp), y0 + (y0 - yp))
                self.curveTo(xi, yi, x2, y2, xn, yn)
            elif op == 'c':
                xp, yp = points[-2:]
                x1, y1, x2, y2, xn, yn = nums
                self.curveTo(xp + x1, yp + y1, xp + x2, yp + y2, xp + xn, yp + yn)
            elif op == 's':
                x2, y2, xn, yn = nums
                if len(points) < 4 or lastop not in {'c', 'C', 's', 'S'}:
                    xp, yp, x0, y0 = points[-2:] * 2
                else:
                    xp, yp, x0, y0 = points[-4:]
                xi, yi = (x0 + (x0 - xp), y0 + (y0 - yp))
                self.curveTo(xi, yi, x0 + x2, y0 + y2, x0 + xn, y0 + yn)
            elif op == 'Q':
                x0, y0 = points[-2:]
                x1, y1, xn, yn = nums
                last_quadratic_cp = (x1, y1)
                (x0, y0), (x1, y1), (x2, y2), (xn, yn) = convert_quadratic_to_cubic_path((x0, y0), (x1, y1), (xn, yn))
                self.curveTo(x1, y1, x2, y2, xn, yn)
            elif op == 'T':
                if last_quadratic_cp is not None:
                    xp, yp = last_quadratic_cp
                else:
                    xp, yp = points[-2:]
                x0, y0 = points[-2:]
                xi, yi = (x0 + (x0 - xp), y0 + (y0 - yp))
                last_quadratic_cp = (xi, yi)
                xn, yn = nums
                (x0, y0), (x1, y1), (x2, y2), (xn, yn) = convert_quadratic_to_cubic_path((x0, y0), (xi, yi), (xn, yn))
                self.curveTo(x1, y1, x2, y2, xn, yn)
            elif op == 'q':
                x0, y0 = points[-2:]
                x1, y1, xn, yn = nums
                x1, y1, xn, yn = (x0 + x1, y0 + y1, x0 + xn, y0 + yn)
                last_quadratic_cp = (x1, y1)
                (x0, y0), (x1, y1), (x2, y2), (xn, yn) = convert_quadratic_to_cubic_path((x0, y0), (x1, y1), (xn, yn))
                self.curveTo(x1, y1, x2, y2, xn, yn)
            elif op == 't':
                if last_quadratic_cp is not None:
                    xp, yp = last_quadratic_cp
                else:
                    xp, yp = points[-2:]
                x0, y0 = points[-2:]
                xn, yn = nums
                xn, yn = (x0 + xn, y0 + yn)
                xi, yi = (x0 + (x0 - xp), y0 + (y0 - yp))
                last_quadratic_cp = (xi, yi)
                (x0, y0), (x1, y1), (x2, y2), (xn, yn) = convert_quadratic_to_cubic_path((x0, y0), (xi, yi), (xn, yn))
                self.curveTo(x1, y1, x2, y2, xn, yn)
            elif op in ('A', 'a'):
                rx, ry, phi, fA, fS, x2, y2 = nums
                x1, y1 = points[-2:]
                if op == 'a':
                    x2 += x1
                    y2 += y1
                if abs(rx) <= 1e-10 or abs(ry) <= 1e-10:
                    self.lineTo(x2, y2)
                else:
                    bp = bezier_arc_from_end_points(x1, y1, rx, ry, phi, fA, fS, x2, y2)
                    for _, _, x1, y1, x2, y2, xn, yn in bp:
                        self.curveTo(x1, y1, x2, y2, xn, yn)
            elif op in ('Z', 'z'):
                self.closePath()
            else:
                logger.debug('Suspicious self operator: %s', op)
            if op not in ('Q', 'q', 'T', 't'):
                last_quadratic_cp = None
            lastop = op
        if self.operators[-1] != _CLOSEPATH:
            unclosed_subpath_pointers.append(len(self.operators))
        if vswap or hswap:
            b = self.getBounds()
            if hswap:
                m = b[2] + b[0]
                for i in range(0, len(points), 2):
                    points[i] = m - points[i]
            if vswap:
                m = b[3] + b[1]
                for i in range(1, len(points), 2):
                    points[i] = m - points[i]
        if unclosed_subpath_pointers and self.fillColor is not None:
            closed_path = Path()
            closed_path.__dict__.update(copy.deepcopy(self.__dict__))
            for pointer in reversed(unclosed_subpath_pointers):
                closed_path.operators.insert(pointer, _CLOSEPATH)
            self.__closed_path = closed_path
            self.fillColor = None
        else:
            self.__closed_path = None

    def provideNode(self):
        p = Path()
        p.__dict__ = self.__dict__.copy()
        del p._SvgPath__closed_path
        if self.__closed_path:
            g = Group()
            g.add(self.__closed_path)
            g.add(p)
            return g
        else:
            return p