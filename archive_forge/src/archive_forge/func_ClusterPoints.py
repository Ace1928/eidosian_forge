import pickle
import numpy
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols, MolSimilarity
from rdkit.ML.Cluster import Murtagh
def ClusterPoints(data, metric, algorithmId, haveLabels=False, haveActs=True, returnDistances=False):
    message('Generating distance matrix.\n')
    dMat = GetDistanceMatrix(data, metric)
    message('Clustering\n')
    clustTree = Murtagh.ClusterData(dMat, len(data), algorithmId, isDistData=1)[0]
    acts = []
    if haveActs and len(data[0]) > 2:
        acts = [int(x[2]) for x in data]
    if not haveLabels:
        labels = [f'Mol: {x[0]}' for x in data]
    else:
        labels = [x[0] for x in data]
    clustTree._ptLabels = labels
    if acts:
        clustTree._ptValues = acts
    for pt in clustTree.GetPoints():
        idx = pt.GetIndex() - 1
        pt.SetName(labels[idx])
        if acts:
            try:
                pt.SetData(int(acts[idx]))
            except Exception:
                pass
    if not returnDistances:
        return clustTree
    return (clustTree, dMat)