from fontTools.cffLib import maxStackLimit
def _convertBlendOpToArgs(blendList):
    if any([isinstance(arg, list) for arg in blendList]):
        args = [i for e in blendList for i in (_convertBlendOpToArgs(e) if isinstance(e, list) else [e])]
    else:
        args = blendList
    numBlends = args[-1]
    args = args[:-1]
    numRegions = len(args) // numBlends - 1
    if not numBlends * (numRegions + 1) == len(args):
        raise ValueError(blendList)
    defaultArgs = [[arg] for arg in args[:numBlends]]
    deltaArgs = args[numBlends:]
    numDeltaValues = len(deltaArgs)
    deltaList = [deltaArgs[i:i + numRegions] for i in range(0, numDeltaValues, numRegions)]
    blend_args = [a + b + [1] for a, b in zip(defaultArgs, deltaList)]
    return blend_args