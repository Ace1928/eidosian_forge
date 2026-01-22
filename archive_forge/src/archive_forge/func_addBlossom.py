from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def addBlossom(base, v, w):
    bb = inblossom[base]
    bv = inblossom[v]
    bw = inblossom[w]
    b = Blossom()
    blossombase[b] = base
    blossomparent[b] = None
    blossomparent[bb] = b
    b.childs = path = []
    b.edges = edgs = [(v, w)]
    while bv != bb:
        blossomparent[bv] = b
        path.append(bv)
        edgs.append(labeledge[bv])
        assert label[bv] == 2 or (label[bv] == 1 and labeledge[bv][0] == mate[blossombase[bv]])
        v = labeledge[bv][0]
        bv = inblossom[v]
    path.append(bb)
    path.reverse()
    edgs.reverse()
    while bw != bb:
        blossomparent[bw] = b
        path.append(bw)
        edgs.append((labeledge[bw][1], labeledge[bw][0]))
        assert label[bw] == 2 or (label[bw] == 1 and labeledge[bw][0] == mate[blossombase[bw]])
        w = labeledge[bw][0]
        bw = inblossom[w]
    assert label[bb] == 1
    label[b] = 1
    labeledge[b] = labeledge[bb]
    blossomdual[b] = 0
    for v in b.leaves():
        if label[inblossom[v]] == 2:
            queue.append(v)
        inblossom[v] = b
    bestedgeto = {}
    for bv in path:
        if isinstance(bv, Blossom):
            if bv.mybestedges is not None:
                nblist = bv.mybestedges
                bv.mybestedges = None
            else:
                nblist = [(v, w) for v in bv.leaves() for w in G.neighbors(v) if v != w]
        else:
            nblist = [(bv, w) for w in G.neighbors(bv) if bv != w]
        for k in nblist:
            i, j = k
            if inblossom[j] == b:
                i, j = (j, i)
            bj = inblossom[j]
            if bj != b and label.get(bj) == 1 and (bj not in bestedgeto or slack(i, j) < slack(*bestedgeto[bj])):
                bestedgeto[bj] = k
        bestedge[bv] = None
    b.mybestedges = list(bestedgeto.values())
    mybestedge = None
    bestedge[b] = None
    for k in b.mybestedges:
        kslack = slack(*k)
        if mybestedge is None or kslack < mybestslack:
            mybestedge = k
            mybestslack = kslack
    bestedge[b] = mybestedge