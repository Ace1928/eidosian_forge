from rdkit.Chem.Pharm2D import SigFactory, Utils
from rdkit.RDLogger import logger
def Gen2DFingerprint(mol, sigFactory, perms=None, dMat=None, bitInfo=None):
    """ generates a 2D fingerprint for a molecule using the
   parameters in _sig_

   **Arguments**

     - mol: the molecule for which the signature should be generated

     - sigFactory : the SigFactory object with signature parameters
       NOTE: no preprocessing is carried out for _sigFactory_.
             It *must* be pre-initialized.

     - perms: (optional) a sequence of permutation indices limiting which
       pharmacophore combinations are allowed

     - dMat: (optional) the distance matrix to be used

     - bitInfo: (optional) used to return the atoms involved in the bits

  """
    if not isinstance(sigFactory, SigFactory.SigFactory):
        raise ValueError('bad factory')
    featFamilies = sigFactory.GetFeatFamilies()
    if _verbose:
        print('* feat famillies:', featFamilies)
    nFeats = len(featFamilies)
    minCount = sigFactory.minPointCount
    maxCount = sigFactory.maxPointCount
    if maxCount > 3:
        logger.warning(' Pharmacophores with more than 3 points are not currently supported.\n' + 'Setting maxCount to 3.')
        maxCount = 3
    if dMat is None:
        from rdkit import Chem
        dMat = Chem.GetDistanceMatrix(mol, sigFactory.includeBondOrder)
    if perms is None:
        perms = []
        for count in range(minCount, maxCount + 1):
            perms.extend(Utils.GetIndexCombinations(nFeats, count))
    featMatches = sigFactory.GetMolFeats(mol)
    if _verbose:
        print('  featMatches:', featMatches)
    sig = sigFactory.GetSignature()
    for perm in perms:
        featClasses = [0] * len(perm)
        for i in range(1, len(perm)):
            if perm[i] == perm[i - 1]:
                featClasses[i] = featClasses[i - 1]
            else:
                featClasses[i] = featClasses[i - 1] + 1
        matchPerms = [featMatches[x] for x in perm]
        if _verbose:
            print(f'\n->Perm: {str(perm)}')
            print(f'    matchPerms: {str(matchPerms)}')
        matchesToMap = Utils.GetUniqueCombinations(matchPerms, featClasses)
        for i, entry in enumerate(matchesToMap):
            matchesToMap[i] = [x[1] for x in entry]
        if _verbose:
            print('    mtM:', matchesToMap)
        for match in matchesToMap:
            if sigFactory.shortestPathsOnly:
                idx = _ShortestPathsMatch(match, perm, sig, dMat, sigFactory)
                if idx is not None and bitInfo is not None:
                    l = bitInfo.get(idx, [])
                    l.append(match)
                    bitInfo[idx] = l
    return sig