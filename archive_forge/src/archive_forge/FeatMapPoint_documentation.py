from rdkit.Chem import ChemicalFeatures

    >>> from rdkit import Geometry
    >>> sfeat = ChemicalFeatures.FreeChemicalFeature('Aromatic','Foo',Geometry.Point3D(0,0,0))
    >>> fmp = FeatMapPoint()
    >>> fmp.initFromFeat(sfeat)
    >>> fmp.GetDirMatch(sfeat)
    1.0

    >>> sfeat.featDirs=[Geometry.Point3D(0,0,1),Geometry.Point3D(0,0,-1)]
    >>> fmp.featDirs=[Geometry.Point3D(0,0,1),Geometry.Point3D(1,0,0)]
    >>> fmp.GetDirMatch(sfeat)
    1.0
    >>> fmp.GetDirMatch(sfeat,useBest=True)
    1.0
    >>> fmp.GetDirMatch(sfeat,useBest=False)
    0.0

    >>> sfeat.featDirs=[Geometry.Point3D(0,0,1)]
    >>> fmp.GetDirMatch(sfeat,useBest=False)
    0.5

    >>> sfeat.featDirs=[Geometry.Point3D(0,0,1)]
    >>> fmp.featDirs=[Geometry.Point3D(0,0,-1)]
    >>> fmp.GetDirMatch(sfeat)
    -1.0
    >>> fmp.GetDirMatch(sfeat,useBest=False)
    -1.0


    