def _displaySubshapeSkelPt(viewer, skelPt, cgoNm, color):
    viewer.server.sphere(tuple(skelPt.location), 0.5, color, cgoNm)
    if hasattr(skelPt, 'shapeDirs'):
        momBeg = skelPt.location - skelPt.shapeDirs[0]
        momEnd = skelPt.location + skelPt.shapeDirs[0]
        viewer.server.cylinder(tuple(momBeg), tuple(momEnd), 0.1, color, cgoNm)