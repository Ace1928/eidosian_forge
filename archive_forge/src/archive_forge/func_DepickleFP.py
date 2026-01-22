import pickle
from rdkit import Chem, DataStructs
def DepickleFP(pkl, similarityMethod):
    if not isinstance(pkl, (bytes, str)):
        pkl = str(pkl)
    try:
        klass = similarityMethods[similarityMethod]
        fp = klass(pkl)
    except Exception:
        import traceback
        traceback.print_exc()
        fp = pickle.loads(pkl)
    return fp