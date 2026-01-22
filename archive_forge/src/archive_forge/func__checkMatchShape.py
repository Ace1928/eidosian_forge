import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def _checkMatchShape(self, targetMol, target, queryMol, query, alignment, builder, targetConf, queryConf, pruneStats=None, tConfId=1001):
    matchOk = True
    TransformMol(queryMol, alignment.transform, confId=queryConf, newConfId=tConfId)
    oSpace = builder.gridSpacing
    builder.gridSpacing = oSpace * 2
    coarseGrid = builder.GenerateSubshapeShape(queryMol, tConfId, addSkeleton=False)
    d = GetShapeShapeDistance(coarseGrid, target.coarseGrid, self.distMetric)
    if d > self.shapeDistTol * self.coarseGridToleranceMult:
        matchOk = False
        if pruneStats is not None:
            pruneStats['coarseGrid'] = pruneStats.get('coarseGrid', 0) + 1
    else:
        builder.gridSpacing = oSpace * 1.5
        medGrid = builder.GenerateSubshapeShape(queryMol, tConfId, addSkeleton=False)
        d = GetShapeShapeDistance(medGrid, target.medGrid, self.distMetric)
        if d > self.shapeDistTol * self.medGridToleranceMult:
            matchOk = False
            if pruneStats is not None:
                pruneStats['medGrid'] = pruneStats.get('medGrid', 0) + 1
        else:
            builder.gridSpacing = oSpace
            fineGrid = builder.GenerateSubshapeShape(queryMol, tConfId, addSkeleton=False)
            d = GetShapeShapeDistance(fineGrid, target, self.distMetric)
            if d > self.shapeDistTol:
                matchOk = False
                if pruneStats is not None:
                    pruneStats['fineGrid'] = pruneStats.get('fineGrid', 0) + 1
            alignment.shapeDist = d
    queryMol.RemoveConformer(tConfId)
    builder.gridSpacing = oSpace
    return matchOk