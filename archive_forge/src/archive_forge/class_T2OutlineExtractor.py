from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class T2OutlineExtractor(T2WidthExtractor):

    def __init__(self, pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private=None, blender=None):
        T2WidthExtractor.__init__(self, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private, blender)
        self.pen = pen
        self.subrLevel = 0

    def reset(self):
        T2WidthExtractor.reset(self)
        self.currentPoint = (0, 0)
        self.sawMoveTo = 0
        self.subrLevel = 0

    def execute(self, charString):
        self.subrLevel += 1
        super().execute(charString)
        self.subrLevel -= 1
        if self.subrLevel == 0:
            self.endPath()

    def _nextPoint(self, point):
        x, y = self.currentPoint
        point = (x + point[0], y + point[1])
        self.currentPoint = point
        return point

    def rMoveTo(self, point):
        self.pen.moveTo(self._nextPoint(point))
        self.sawMoveTo = 1

    def rLineTo(self, point):
        if not self.sawMoveTo:
            self.rMoveTo((0, 0))
        self.pen.lineTo(self._nextPoint(point))

    def rCurveTo(self, pt1, pt2, pt3):
        if not self.sawMoveTo:
            self.rMoveTo((0, 0))
        nextPoint = self._nextPoint
        self.pen.curveTo(nextPoint(pt1), nextPoint(pt2), nextPoint(pt3))

    def closePath(self):
        if self.sawMoveTo:
            self.pen.closePath()
        self.sawMoveTo = 0

    def endPath(self):
        if self.sawMoveTo:
            self.closePath()

    def op_rmoveto(self, index):
        self.endPath()
        self.rMoveTo(self.popallWidth())

    def op_hmoveto(self, index):
        self.endPath()
        self.rMoveTo((self.popallWidth(1)[0], 0))

    def op_vmoveto(self, index):
        self.endPath()
        self.rMoveTo((0, self.popallWidth(1)[0]))

    def op_endchar(self, index):
        self.endPath()
        args = self.popallWidth()
        if args:
            from fontTools.encodings.StandardEncoding import StandardEncoding
            adx, ady, bchar, achar = args
            baseGlyph = StandardEncoding[bchar]
            self.pen.addComponent(baseGlyph, (1, 0, 0, 1, 0, 0))
            accentGlyph = StandardEncoding[achar]
            self.pen.addComponent(accentGlyph, (1, 0, 0, 1, adx, ady))

    def op_rlineto(self, index):
        args = self.popall()
        for i in range(0, len(args), 2):
            point = args[i:i + 2]
            self.rLineTo(point)

    def op_hlineto(self, index):
        self.alternatingLineto(1)

    def op_vlineto(self, index):
        self.alternatingLineto(0)

    def op_rrcurveto(self, index):
        """{dxa dya dxb dyb dxc dyc}+ rrcurveto"""
        args = self.popall()
        for i in range(0, len(args), 6):
            dxa, dya, dxb, dyb, dxc, dyc = args[i:i + 6]
            self.rCurveTo((dxa, dya), (dxb, dyb), (dxc, dyc))

    def op_rcurveline(self, index):
        """{dxa dya dxb dyb dxc dyc}+ dxd dyd rcurveline"""
        args = self.popall()
        for i in range(0, len(args) - 2, 6):
            dxb, dyb, dxc, dyc, dxd, dyd = args[i:i + 6]
            self.rCurveTo((dxb, dyb), (dxc, dyc), (dxd, dyd))
        self.rLineTo(args[-2:])

    def op_rlinecurve(self, index):
        """{dxa dya}+ dxb dyb dxc dyc dxd dyd rlinecurve"""
        args = self.popall()
        lineArgs = args[:-6]
        for i in range(0, len(lineArgs), 2):
            self.rLineTo(lineArgs[i:i + 2])
        dxb, dyb, dxc, dyc, dxd, dyd = args[-6:]
        self.rCurveTo((dxb, dyb), (dxc, dyc), (dxd, dyd))

    def op_vvcurveto(self, index):
        """dx1? {dya dxb dyb dyc}+ vvcurveto"""
        args = self.popall()
        if len(args) % 2:
            dx1 = args[0]
            args = args[1:]
        else:
            dx1 = 0
        for i in range(0, len(args), 4):
            dya, dxb, dyb, dyc = args[i:i + 4]
            self.rCurveTo((dx1, dya), (dxb, dyb), (0, dyc))
            dx1 = 0

    def op_hhcurveto(self, index):
        """dy1? {dxa dxb dyb dxc}+ hhcurveto"""
        args = self.popall()
        if len(args) % 2:
            dy1 = args[0]
            args = args[1:]
        else:
            dy1 = 0
        for i in range(0, len(args), 4):
            dxa, dxb, dyb, dxc = args[i:i + 4]
            self.rCurveTo((dxa, dy1), (dxb, dyb), (dxc, 0))
            dy1 = 0

    def op_vhcurveto(self, index):
        """dy1 dx2 dy2 dx3 {dxa dxb dyb dyc dyd dxe dye dxf}* dyf? vhcurveto (30)
        {dya dxb dyb dxc dxd dxe dye dyf}+ dxf? vhcurveto
        """
        args = self.popall()
        while args:
            args = self.vcurveto(args)
            if args:
                args = self.hcurveto(args)

    def op_hvcurveto(self, index):
        """dx1 dx2 dy2 dy3 {dya dxb dyb dxc dxd dxe dye dyf}* dxf?
        {dxa dxb dyb dyc dyd dxe dye dxf}+ dyf?
        """
        args = self.popall()
        while args:
            args = self.hcurveto(args)
            if args:
                args = self.vcurveto(args)

    def op_hflex(self, index):
        dx1, dx2, dy2, dx3, dx4, dx5, dx6 = self.popall()
        dy1 = dy3 = dy4 = dy6 = 0
        dy5 = -dy2
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_flex(self, index):
        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, dx6, dy6, fd = self.popall()
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_hflex1(self, index):
        dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = self.popall()
        dy3 = dy4 = 0
        dy6 = -(dy1 + dy2 + dy3 + dy4 + dy5)
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_flex1(self, index):
        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = self.popall()
        dx = dx1 + dx2 + dx3 + dx4 + dx5
        dy = dy1 + dy2 + dy3 + dy4 + dy5
        if abs(dx) > abs(dy):
            dx6 = d6
            dy6 = -dy
        else:
            dx6 = -dx
            dy6 = d6
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_and(self, index):
        raise NotImplementedError

    def op_or(self, index):
        raise NotImplementedError

    def op_not(self, index):
        raise NotImplementedError

    def op_store(self, index):
        raise NotImplementedError

    def op_abs(self, index):
        raise NotImplementedError

    def op_add(self, index):
        raise NotImplementedError

    def op_sub(self, index):
        raise NotImplementedError

    def op_div(self, index):
        num2 = self.pop()
        num1 = self.pop()
        d1 = num1 // num2
        d2 = num1 / num2
        if d1 == d2:
            self.push(d1)
        else:
            self.push(d2)

    def op_load(self, index):
        raise NotImplementedError

    def op_neg(self, index):
        raise NotImplementedError

    def op_eq(self, index):
        raise NotImplementedError

    def op_drop(self, index):
        raise NotImplementedError

    def op_put(self, index):
        raise NotImplementedError

    def op_get(self, index):
        raise NotImplementedError

    def op_ifelse(self, index):
        raise NotImplementedError

    def op_random(self, index):
        raise NotImplementedError

    def op_mul(self, index):
        raise NotImplementedError

    def op_sqrt(self, index):
        raise NotImplementedError

    def op_dup(self, index):
        raise NotImplementedError

    def op_exch(self, index):
        raise NotImplementedError

    def op_index(self, index):
        raise NotImplementedError

    def op_roll(self, index):
        raise NotImplementedError

    def alternatingLineto(self, isHorizontal):
        args = self.popall()
        for arg in args:
            if isHorizontal:
                point = (arg, 0)
            else:
                point = (0, arg)
            self.rLineTo(point)
            isHorizontal = not isHorizontal

    def vcurveto(self, args):
        dya, dxb, dyb, dxc = args[:4]
        args = args[4:]
        if len(args) == 1:
            dyc = args[0]
            args = []
        else:
            dyc = 0
        self.rCurveTo((0, dya), (dxb, dyb), (dxc, dyc))
        return args

    def hcurveto(self, args):
        dxa, dxb, dyb, dyc = args[:4]
        args = args[4:]
        if len(args) == 1:
            dxc = args[0]
            args = []
        else:
            dxc = 0
        self.rCurveTo((dxa, 0), (dxb, dyb), (dxc, dyc))
        return args