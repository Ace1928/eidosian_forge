def GetPointsPositions(self):
    if self._pointsPositions is not None:
        return self._pointsPositions
    else:
        self._GenPoints()
        return self._pointsPositions