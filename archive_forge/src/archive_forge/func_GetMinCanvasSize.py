from sping import pid as piddle
def GetMinCanvasSize(adjList, levelList):
    maxAcross = -1
    for k in levelList.keys():
        nHere = len(levelList[k])
        maxAcross = max(maxAcross, nHere)
    nLevs = len(levelList.keys())
    minSize = (maxAcross * (visOpts.minCircRad * 2 + visOpts.horizOffset), visOpts.topMargin + nLevs * visOpts.vertOffset)
    return minSize