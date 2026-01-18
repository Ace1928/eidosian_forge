from .interpolatableHelpers import *
def test_starting_point(glyph0, glyph1, ix, tolerance, matching):
    if matching is None:
        matching = list(range(len(glyph0.isomorphisms)))
    contour0 = glyph0.isomorphisms[ix]
    contour1 = glyph1.isomorphisms[matching[ix]]
    m0Vectors = glyph0.greenVectors
    m1Vectors = [glyph1.greenVectors[i] for i in matching]
    c0 = contour0[0]
    costs = [vdiff_hypot2_complex(c0[0], c1[0]) for c1 in contour1]
    min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
    first_cost = costs[0]
    proposed_point = contour1[min_cost_idx][1]
    reverse = contour1[min_cost_idx][2]
    if min_cost < first_cost * tolerance:
        num_points = len(glyph1.points[ix])
        leeway = 3
        if not reverse and (proposed_point <= leeway or proposed_point >= num_points - leeway):
            transforms = []
            for vector in (m0Vectors[ix], m1Vectors[ix]):
                meanX = vector[1]
                meanY = vector[2]
                stddevX = vector[3] * 0.5
                stddevY = vector[4] * 0.5
                correlation = vector[5] / abs(vector[0])
                a = stddevX * stddevX
                c = stddevY * stddevY
                b = correlation * stddevX * stddevY
                delta = (((a - c) * 0.5) ** 2 + b * b) ** 0.5
                lambda1 = (a + c) * 0.5 + delta
                lambda2 = (a + c) * 0.5 - delta
                theta = atan2(lambda1 - a, b) if b != 0 else pi * 0.5 if a < c else 0
                trans = Transform()
                trans = trans.rotate(theta)
                trans = trans.scale(sqrt(lambda1), sqrt(lambda2))
                transforms.append(trans)
            trans = transforms[0]
            new_c0 = ([complex(*trans.transformPoint((pt.real, pt.imag))) for pt in c0[0]],) + c0[1:]
            trans = transforms[1]
            new_contour1 = []
            for c1 in contour1:
                new_c1 = ([complex(*trans.transformPoint((pt.real, pt.imag))) for pt in c1[0]],) + c1[1:]
                new_contour1.append(new_c1)
            costs = [vdiff_hypot2_complex(new_c0[0], new_c1[0]) for new_c1 in new_contour1]
            min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
            first_cost = costs[0]
            if min_cost < first_cost * tolerance:
                pass
    this_tolerance = min_cost / first_cost if first_cost else 1
    log.debug('test-starting-point: tolerance %g', this_tolerance)
    return (this_tolerance, proposed_point, reverse)