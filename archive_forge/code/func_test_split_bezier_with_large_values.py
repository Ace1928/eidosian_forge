from matplotlib.bezier import inside_circle, split_bezier_intersecting_with_closedpath
def test_split_bezier_with_large_values():
    arrow_path = [(96950809781500.0, 804.7503795623779), (96950809781500.0, 859.6242585800646), (96950809781500.0, 914.4981375977513)]
    in_f = inside_circle(96950809781500.0, 804.7503795623779, 0.06)
    split_bezier_intersecting_with_closedpath(arrow_path, in_f)