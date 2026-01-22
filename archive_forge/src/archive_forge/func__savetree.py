import numbers
from . import _cluster  # type: ignore
def _savetree(self, jobname, tree, order, transpose):
    """Save the hierarchical clustering solution (PRIVATE)."""
    if transpose:
        extension = '.atr'
        keyword = 'ARRY'
    else:
        extension = '.gtr'
        keyword = 'GENE'
    index = tree.sort(order)
    nnodes = len(tree)
    with open(jobname + extension, 'w') as outputfile:
        nodeID = [''] * nnodes
        nodedist = np.array([node.distance for node in tree[:]])
        for nodeindex in range(nnodes):
            min1 = tree[nodeindex].left
            min2 = tree[nodeindex].right
            nodeID[nodeindex] = 'NODE%dX' % (nodeindex + 1)
            outputfile.write(nodeID[nodeindex])
            outputfile.write('\t')
            if min1 < 0:
                index1 = -min1 - 1
                outputfile.write(nodeID[index1] + '\t')
                nodedist[nodeindex] = max(nodedist[nodeindex], nodedist[index1])
            else:
                outputfile.write('%s%dX\t' % (keyword, min1))
            if min2 < 0:
                index2 = -min2 - 1
                outputfile.write(nodeID[index2] + '\t')
                nodedist[nodeindex] = max(nodedist[nodeindex], nodedist[index2])
            else:
                outputfile.write('%s%dX\t' % (keyword, min2))
            outputfile.write(str(1.0 - nodedist[nodeindex]))
            outputfile.write('\n')
    return index