import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def CalculateDirectionsAtPoint(pt, shapeGrid, winRad):
    shapeGridVect = shapeGrid.GetOccupancyVect()
    nGridPts = len(shapeGridVect)
    tmp = winRad / shapeGrid.GetSpacing()
    radInGrid = int(tmp)
    radInGrid2 = int(tmp * tmp)
    covMat = numpy.zeros((3, 3), numpy.float64)
    dX = shapeGrid.GetNumX()
    dY = shapeGrid.GetNumY()
    idx = shapeGrid.GetGridPointIndex(pt.location)
    idxZ = idx // (dX * dY)
    rem = idx % (dX * dY)
    idxY = rem // dX
    idxX = rem % dX
    totWt = 0.0
    for i in range(-radInGrid, radInGrid):
        xi = idxX + i
        for j in range(-radInGrid, radInGrid):
            xj = idxY + j
            for k in range(-radInGrid, radInGrid):
                xk = idxZ + k
                d2 = i * i + j * j + k * k
                if d2 > radInGrid2 and int(math.sqrt(d2)) > radInGrid:
                    continue
                gridIdx = (xk * dY + xj) * dX + xi
                if gridIdx >= 0 and gridIdx < nGridPts:
                    wtHere = shapeGridVect[gridIdx]
                    totWt += wtHere
                    ptInSphere = shapeGrid.GetGridPointLoc(gridIdx)
                    ptInSphere -= pt.location
                    covMat[0][0] += wtHere * (ptInSphere.x * ptInSphere.x)
                    covMat[0][1] += wtHere * (ptInSphere.x * ptInSphere.y)
                    covMat[0][2] += wtHere * (ptInSphere.x * ptInSphere.z)
                    covMat[1][1] += wtHere * (ptInSphere.y * ptInSphere.y)
                    covMat[1][2] += wtHere * (ptInSphere.y * ptInSphere.z)
                    covMat[2][2] += wtHere * (ptInSphere.z * ptInSphere.z)
    covMat /= totWt
    covMat[1][0] = covMat[0][1]
    covMat[2][0] = covMat[0][2]
    covMat[2][1] = covMat[1][2]
    eVals, eVects = numpy.linalg.eigh(covMat)
    sv = list(zip(eVals, numpy.transpose(eVects)))
    sv.sort(reverse=True)
    eVals, eVects = list(zip(*sv))
    pt.shapeMoments = tuple(eVals)
    pt.shapeDirs = tuple([Geometry.Point3D(p[0], p[1], p[2]) for p in eVects])