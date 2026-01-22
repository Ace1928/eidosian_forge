from .interpolatableHelpers import *
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.recordingPen import (
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.cairoPen import CairoPen
from fontTools.pens.pointPen import (
from fontTools.varLib.interpolatableHelpers import (
from itertools import cycle
from functools import wraps
from io import BytesIO
import cairo
import math
import os
import logging
def add_problem(self, glyphname, problems, *, show_tolerance=True, show_page_number=True):
    if type(problems) not in (list, tuple):
        problems = [problems]
    self.toc[self.page_number] = (glyphname, problems)
    problem_type = problems[0]['type']
    problem_types = set((problem['type'] for problem in problems))
    if not all((pt == problem_type for pt in problem_types)):
        problem_type = ', '.join(sorted({problem['type'] for problem in problems}))
    log.info('Drawing %s: %s', glyphname, problem_type)
    master_keys = ('master_idx',) if 'master_idx' in problems[0] else ('master_1_idx', 'master_2_idx')
    master_indices = [problems[0][k] for k in master_keys]
    if problem_type == InterpolatableProblem.MISSING:
        sample_glyph = next((i for i, m in enumerate(self.glyphsets) if m[glyphname] is not None))
        master_indices.insert(0, sample_glyph)
    x = self.pad
    y = self.pad
    self.draw_label('Glyph name: ' + glyphname, x=x, y=y, color=self.head_color, align=0, bold=True, font_size=self.title_font_size)
    tolerance = min((p.get('tolerance', 1) for p in problems))
    if tolerance < 1 and show_tolerance:
        self.draw_label('tolerance: %.2f' % tolerance, x=x, y=y, width=self.width - 2 * self.pad, align=1, bold=True)
    y += self.title_font_size + self.pad
    self.draw_label('Problems: ' + problem_type, x=x, y=y, width=self.width - 2 * self.pad, color=self.head_color, bold=True)
    y += self.font_size + self.pad * 2
    scales = []
    for which, master_idx in enumerate(master_indices):
        glyphset = self.glyphsets[master_idx]
        name = self.names[master_idx]
        self.draw_label(name, x=x, y=y, color=self.label_color, width=self.panel_width, align=0.5)
        y += self.font_size + self.pad
        if glyphset[glyphname] is not None:
            scales.append(self.draw_glyph(glyphset, glyphname, problems, which, x=x, y=y))
        else:
            self.draw_emoticon(self.shrug, x=x, y=y)
        y += self.panel_height + self.font_size + self.pad
    if any((pt in (InterpolatableProblem.NOTHING, InterpolatableProblem.WRONG_START_POINT, InterpolatableProblem.CONTOUR_ORDER, InterpolatableProblem.KINK, InterpolatableProblem.UNDERWEIGHT, InterpolatableProblem.OVERWEIGHT) for pt in problem_types)):
        x = self.pad + self.panel_width + self.pad
        y = self.pad
        y += self.title_font_size + self.pad * 2
        y += self.font_size + self.pad
        glyphset1 = self.glyphsets[master_indices[0]]
        glyphset2 = self.glyphsets[master_indices[1]]
        self.draw_label('midway interpolation', x=x, y=y, color=self.head_color, width=self.panel_width, align=0.5)
        y += self.font_size + self.pad
        midway_glyphset = LerpGlyphSet(glyphset1, glyphset2)
        self.draw_glyph(midway_glyphset, glyphname, [{'type': 'midway'}] + [p for p in problems if p['type'] in (InterpolatableProblem.KINK, InterpolatableProblem.UNDERWEIGHT, InterpolatableProblem.OVERWEIGHT)], None, x=x, y=y, scale=min(scales))
        y += self.panel_height + self.font_size + self.pad
    if any((pt in (InterpolatableProblem.WRONG_START_POINT, InterpolatableProblem.CONTOUR_ORDER, InterpolatableProblem.KINK) for pt in problem_types)):
        self.draw_label('proposed fix', x=x, y=y, color=self.head_color, width=self.panel_width, align=0.5)
        y += self.font_size + self.pad
        overriding1 = OverridingDict(glyphset1)
        overriding2 = OverridingDict(glyphset2)
        perContourPen1 = PerContourOrComponentPen(RecordingPen, glyphset=overriding1)
        perContourPen2 = PerContourOrComponentPen(RecordingPen, glyphset=overriding2)
        glyphset1[glyphname].draw(perContourPen1)
        glyphset2[glyphname].draw(perContourPen2)
        for problem in problems:
            if problem['type'] == InterpolatableProblem.CONTOUR_ORDER:
                fixed_contours = [perContourPen2.value[i] for i in problems[0]['value_2']]
                perContourPen2.value = fixed_contours
        for problem in problems:
            if problem['type'] == InterpolatableProblem.WRONG_START_POINT:
                wrongContour1 = perContourPen1.value[problem['contour']]
                wrongContour2 = perContourPen2.value[problem['contour']]
                points1 = RecordingPointPen()
                converter = SegmentToPointPen(points1, False)
                wrongContour1.replay(converter)
                points2 = RecordingPointPen()
                converter = SegmentToPointPen(points2, False)
                wrongContour2.replay(converter)
                proposed_start = problem['value_2']
                if problem['reversed']:
                    new_points2 = RecordingPointPen()
                    reversedPen = ReverseContourPointPen(new_points2)
                    points2.replay(reversedPen)
                    points2 = new_points2
                    proposed_start = len(points2.value) - 2 - proposed_start
                beginPath = points2.value[:1]
                endPath = points2.value[-1:]
                pts = points2.value[1:-1]
                pts = pts[proposed_start:] + pts[:proposed_start]
                points2.value = beginPath + pts + endPath
                segment1 = RecordingPen()
                converter = PointToSegmentPen(segment1, True)
                points1.replay(converter)
                segment2 = RecordingPen()
                converter = PointToSegmentPen(segment2, True)
                points2.replay(converter)
                wrongContour1.value = segment1.value
                wrongContour2.value = segment2.value
                perContourPen1.value[problem['contour']] = wrongContour1
                perContourPen2.value[problem['contour']] = wrongContour2
        for problem in problems:
            if problem['type'] == InterpolatableProblem.KINK:
                wrongContour1 = perContourPen1.value[problem['contour']]
                wrongContour2 = perContourPen2.value[problem['contour']]
                points1 = RecordingPointPen()
                converter = SegmentToPointPen(points1, False)
                wrongContour1.replay(converter)
                points2 = RecordingPointPen()
                converter = SegmentToPointPen(points2, False)
                wrongContour2.replay(converter)
                i = problem['value']
                j = i + 1
                pt0 = points1.value[j][1][0]
                pt1 = points2.value[j][1][0]
                j_prev = (i - 1) % (len(points1.value) - 2) + 1
                pt0_prev = points1.value[j_prev][1][0]
                pt1_prev = points2.value[j_prev][1][0]
                j_next = (i + 1) % (len(points1.value) - 2) + 1
                pt0_next = points1.value[j_next][1][0]
                pt1_next = points2.value[j_next][1][0]
                pt0 = complex(*pt0)
                pt1 = complex(*pt1)
                pt0_prev = complex(*pt0_prev)
                pt1_prev = complex(*pt1_prev)
                pt0_next = complex(*pt0_next)
                pt1_next = complex(*pt1_next)
                r0 = abs(pt0 - pt0_prev) / abs(pt0_next - pt0_prev)
                r1 = abs(pt1 - pt1_prev) / abs(pt1_next - pt1_prev)
                r_mid = (r0 + r1) / 2
                pt0 = pt0_prev + r_mid * (pt0_next - pt0_prev)
                pt1 = pt1_prev + r_mid * (pt1_next - pt1_prev)
                points1.value[j] = (points1.value[j][0], ((pt0.real, pt0.imag),) + points1.value[j][1][1:], points1.value[j][2])
                points2.value[j] = (points2.value[j][0], ((pt1.real, pt1.imag),) + points2.value[j][1][1:], points2.value[j][2])
                segment1 = RecordingPen()
                converter = PointToSegmentPen(segment1, True)
                points1.replay(converter)
                segment2 = RecordingPen()
                converter = PointToSegmentPen(segment2, True)
                points2.replay(converter)
                wrongContour1.value = segment1.value
                wrongContour2.value = segment2.value
        fixed1 = RecordingPen()
        fixed2 = RecordingPen()
        for contour in perContourPen1.value:
            fixed1.value.extend(contour.value)
        for contour in perContourPen2.value:
            fixed2.value.extend(contour.value)
        fixed1.draw = fixed1.replay
        fixed2.draw = fixed2.replay
        overriding1[glyphname] = fixed1
        overriding2[glyphname] = fixed2
        try:
            midway_glyphset = LerpGlyphSet(overriding1, overriding2)
            self.draw_glyph(midway_glyphset, glyphname, {'type': 'fixed'}, None, x=x, y=y, scale=min(scales))
        except ValueError:
            self.draw_emoticon(self.shrug, x=x, y=y)
        y += self.panel_height + self.pad
    else:
        emoticon = self.shrug
        if InterpolatableProblem.UNDERWEIGHT in problem_types:
            emoticon = self.underweight
        elif InterpolatableProblem.OVERWEIGHT in problem_types:
            emoticon = self.overweight
        elif InterpolatableProblem.NOTHING in problem_types:
            emoticon = self.yay
        self.draw_emoticon(emoticon, x=x, y=y)
    if show_page_number:
        self.draw_label(str(self.page_number), x=0, y=self.height - self.font_size - self.pad, width=self.width, color=self.head_color, align=0.5)