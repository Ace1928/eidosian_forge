import copy, logging
from pyomo.common.dependencies import numpy
class FOQUSGraph(object):

    def solve_tear_direct(self, G, order, function, tears, outEdges, iterLim, tol, tol_type, report_diffs):
        """
        Use direct substitution to solve tears. If multiple tears are
        given they are solved simultaneously.

        Arguments
        ---------
            order
                List of lists of order in which to calculate nodes
            tears
                List of tear edge indexes
            iterLim
                Limit on the number of iterations to run
            tol
                Tolerance at which iteration can be stopped

        Returns
        -------
            list
                List of lists of diff history, differences between input and
                output values at each iteration
        """
        hist = []
        if not len(tears):
            self.run_order(G, order, function, tears)
            return hist
        logger.info('Starting Direct tear convergence')
        ignore = tears + outEdges
        itercount = 0
        while True:
            svals, dvals = self.tear_diff_direct(G, tears)
            err = self.compute_err(svals, dvals, tol_type)
            hist.append(err)
            if report_diffs:
                print('Diff matrix:\n%s' % err)
            if numpy.max(numpy.abs(err)) < tol:
                break
            if itercount >= iterLim:
                logger.warning('Direct failed to converge in %s iterations' % iterLim)
                return hist
            self.pass_tear_direct(G, tears)
            itercount += 1
            logger.info('Running Direct iteration %s' % itercount)
            self.run_order(G, order, function, ignore)
        self.pass_edges(G, outEdges)
        logger.info('Direct converged in %s iterations' % itercount)
        return hist

    def solve_tear_wegstein(self, G, order, function, tears, outEdges, iterLim, tol, tol_type, report_diffs, accel_min, accel_max):
        """
        Use Wegstein to solve tears. If multiple tears are given
        they are solved simultaneously.

        Arguments
        ---------
            order
                List of lists of order in which to calculate nodes
            tears
                List of tear edge indexes
            iterLim
                Limit on the number of iterations to run
            tol
                Tolerance at which iteration can be stopped
            accel_min
                Minimum value for Wegstein acceleration factor
            accel_max
                Maximum value for Wegstein acceleration factor
            tol_type
                Type of tolerance value, either "abs" (absolute) or
                "rel" (relative to current value)

        Returns
        -------
            list
                List of lists of diff history, differences between input and
                output values at each iteration
        """
        hist = []
        if not len(tears):
            self.run_order(G, order, function, tears)
            return hist
        logger.info('Starting Wegstein tear convergence')
        itercount = 0
        ignore = tears + outEdges
        gofx = self.generate_gofx(G, tears)
        x = self.generate_first_x(G, tears)
        err = self.compute_err(gofx, x, tol_type)
        hist.append(err)
        if report_diffs:
            print('Diff matrix:\n%s' % err)
        if numpy.max(numpy.abs(err)) < tol:
            logger.info('Wegstein converged in %s iterations' % itercount)
            return hist
        x_prev = x
        gofx_prev = gofx
        x = gofx
        self.pass_tear_wegstein(G, tears, gofx)
        while True:
            itercount += 1
            logger.info('Running Wegstein iteration %s' % itercount)
            self.run_order(G, order, function, ignore)
            gofx = self.generate_gofx(G, tears)
            err = self.compute_err(gofx, x, tol_type)
            hist.append(err)
            if report_diffs:
                print('Diff matrix:\n%s' % err)
            if numpy.max(numpy.abs(err)) < tol:
                break
            if itercount > iterLim:
                logger.warning('Wegstein failed to converge in %s iterations' % iterLim)
                return hist
            denom = x - x_prev
            old_settings = numpy.seterr(divide='ignore', invalid='ignore')
            slope = numpy.divide(gofx - gofx_prev, denom)
            numpy.seterr(**old_settings)
            slope[numpy.isnan(slope)] = 0
            slope[numpy.isinf(slope)] = 0
            accel = slope / (slope - 1)
            accel[accel < accel_min] = accel_min
            accel[accel > accel_max] = accel_max
            x_prev = x
            gofx_prev = gofx
            x = accel * x_prev + (1 - accel) * gofx_prev
            self.pass_tear_wegstein(G, tears, x)
        self.pass_edges(G, outEdges)
        logger.info('Wegstein converged in %s iterations' % itercount)
        return hist

    def scc_collect(self, G, excludeEdges=None):
        """
        This is an algorithm for finding strongly connected components (SCCs)
        in a graph. It is based on Tarjan. 1972 Depth-First Search and Linear
        Graph Algorithms, SIAM J. Comput. v1 no. 2 1972

        Returns
        -------
            sccNodes
                List of lists of nodes in each SCC
            sccEdges
                List of lists of edge indexes in each SCC
            sccOrder
                List of lists for order in which to calculate SCCs
            outEdges
                List of lists of edge indexes leaving the SCC
        """

        def sc(v, stk, depth, stringComps):
            ndepth[v] = depth
            back[v] = depth
            depth += 1
            stk.append(v)
            for w in adj[v]:
                if ndepth[w] == None:
                    sc(w, stk, depth, stringComps)
                    back[v] = min(back[w], back[v])
                elif w in stk:
                    back[v] = min(back[w], back[v])
            if back[v] == ndepth[v]:
                scomp = []
                while True:
                    w = stk.pop()
                    scomp.append(i2n[w])
                    if w == v:
                        break
                stringComps.append(scomp)
            return depth
        i2n, adj, _ = self.adj_lists(G, excludeEdges=excludeEdges)
        stk = []
        stringComps = []
        ndepth = [None] * len(i2n)
        back = [None] * len(i2n)
        for v in range(len(i2n)):
            if ndepth[v] == None:
                sc(v, stk, 0, stringComps)
        sccNodes = stringComps
        sccEdges = []
        outEdges = []
        inEdges = []
        for nset in stringComps:
            e, ie, oe = self.sub_graph_edges(G, nset)
            sccEdges.append(e)
            inEdges.append(ie)
            outEdges.append(oe)
        sccOrder = self.scc_calculation_order(sccNodes, inEdges, outEdges)
        return (sccNodes, sccEdges, sccOrder, outEdges)

    def scc_calculation_order(self, sccNodes, ie, oe):
        """
        This determines the order in which to do calculations for strongly
        connected components. It is used to help determine the most efficient
        order to solve tear streams to prevent extra iterations. This just
        makes an adjacency list with the SCCs as nodes and calls the tree
        order function.

        Arguments
        ---------
            sccNodes
                List of lists of nodes in each SCC
            ie
                List of lists of in edge indexes to SCCs
            oe
                List of lists of out edge indexes to SCCs

        """
        adj = []
        adjR = []
        for i in range(len(sccNodes)):
            adj.append([])
            adjR.append([])
        done = False
        for i in range(len(sccNodes)):
            for j in range(len(sccNodes)):
                for ine in ie[i]:
                    for oute in oe[j]:
                        if ine == oute:
                            adj[j].append(i)
                            adjR[i].append(j)
                            done = True
                    if done:
                        break
                if done:
                    break
            done = False
        return self.tree_order(adj, adjR)

    def calculation_order(self, G, roots=None, nodes=None):
        """
        Rely on tree_order to return a calculation order of nodes

        Arguments
        ---------
            roots
                List of nodes to consider as tree roots,
                if None then the actual roots are used
            nodes
                Subset of nodes to consider in the tree,
                if None then all nodes are used
        """
        tset = self.tear_set(G)
        i2n, adj, adjR = self.adj_lists(G, excludeEdges=tset, nodes=nodes)
        order = []
        if roots is not None:
            node_map = self.node_to_idx(G)
            rootsIndex = []
            for node in roots:
                rootsIndex.append(node_map[node])
        else:
            rootsIndex = None
        orderIndex = self.tree_order(adj, adjR, rootsIndex)
        for i in range(len(orderIndex)):
            order.append([])
            for j in range(len(orderIndex[i])):
                order[i].append(i2n[orderIndex[i][j]])
        return order

    def tree_order(self, adj, adjR, roots=None):
        """
        This function determines the ordering of nodes in a directed
        tree. This is a generic function that can operate on any
        given tree represented by the adjaceny and reverse
        adjacency lists. If the adjacency list does not represent
        a tree the results are not valid.

        In the returned order, it is sometimes possible for more
        than one node to be calculated at once. So a list of lists
        is returned by this function. These represent a bredth
        first search order of the tree. Following the order, all
        nodes that lead to a particular node will be visited
        before it.

        Arguments
        ---------
            adj
                An adjeceny list for a directed tree. This uses
                generic integer node indexes, not node names from the
                graph itself. This allows this to be used on sub-graphs
                and graps of components more easily.
            adjR
                The reverse adjacency list coresponing to adj
            roots
                List of node indexes to start from. These do not
                need to be the root nodes of the tree, in some cases
                like when a node changes the changes may only affect
                nodes reachable in the tree from the changed node, in
                the case that roots are supplied not all the nodes in
                the tree may appear in the ordering. If no roots are
                supplied, the roots of the tree are used.
        """
        adjR = copy.deepcopy(adjR)
        for i, l in enumerate(adjR):
            adjR[i] = set(l)
        if roots is None:
            roots = []
            mark = [True] * len(adj)
            r = [True] * len(adj)
            for sucs in adj:
                for i in sucs:
                    r[i] = False
            for i in range(len(r)):
                if r[i]:
                    roots.append(i)
        else:
            mark = [False] * len(adj)
            lst = roots
            while len(lst) > 0:
                lst2 = []
                for i in lst:
                    mark[i] = True
                    lst2 += adj[i]
                lst = set(lst2)
        ndepth = [None] * len(adj)
        lst = copy.deepcopy(roots)
        order = []
        checknodes = set()
        for i in roots:
            checknodes.update(adj[i])
        depth = 0
        while len(lst) > 0:
            order.append(lst)
            depth += 1
            lst = []
            delSet = set()
            checkUpdate = set()
            for i in checknodes:
                if ndepth[i] != None:
                    raise RuntimeError('Function tree_order does not work with cycles')
                remSet = set()
                for j in adjR[i]:
                    if j in order[depth - 1]:
                        remSet.add(j)
                    elif mark[j] == False:
                        remSet.add(j)
                adjR[i] = adjR[i].difference(remSet)
                if len(adjR[i]) == 0:
                    ndepth[i] = depth
                    lst.append(i)
                    delSet.add(i)
                    checkUpdate.update(adj[i])
            checknodes = checknodes.difference(delSet)
            checknodes = checknodes.union(checkUpdate)
        return order

    def check_tear_set(self, G, tset):
        """
        Check whether the specified tear streams are sufficient.
        If the graph minus the tear edges is not a tree then the
        tear set is not sufficient to solve the graph.
        """
        sccNodes, _, _, _ = self.scc_collect(G, excludeEdges=tset)
        for nodes in sccNodes:
            if len(nodes) > 1:
                return False
        return True

    def select_tear_heuristic(self, G):
        """
        This finds optimal sets of tear edges based on two criteria.
        The primary objective is to minimize the maximum number of
        times any cycle is broken. The secondary criteria is to
        minimize the number of tears.

        This function uses a branch and bound type approach.

        Returns
        -------
            tsets
                List of lists of tear sets. All the tear sets returned
                are equally good. There are often a very large number
                of equally good tear sets.
            upperbound_loop
                The max number of times any single loop is torn
            upperbound_total
                The total number of loops

        Improvements for the future

        I think I can improve the efficiency of this, but it is good
        enough for now. Here are some ideas for improvement:

            1. Reduce the number of redundant solutions. It is possible
            to find tears sets [1,2] and [2,1]. I eliminate
            redundant solutions from the results, but they can
            occur and it reduces efficiency.

            2. Look at strongly connected components instead of whole
            graph. This would cut back on the size of graph we are
            looking at. The flowsheets are rarely one strongly
            connected component.

            3. When you add an edge to a tear set you could reduce the
            size of the problem in the branch by only looking at
            strongly connected components with that edge removed.

            4. This returns all equally good optimal tear sets. That
            may not really be necessary. For very large flowsheets,
            there could be an extremely large number of optimal tear
            edge sets.
        """

        def sear(depth, prevY):
            for i in range(len(cycleEdges[depth])):
                y = list(prevY)
                y[cycleEdges[depth][i]] = 1
                Ay = numpy.dot(A, y)
                maxAy = max(Ay)
                sumY = sum(y)
                if maxAy > upperBound[0]:
                    continue
                elif maxAy == upperBound[0] and sumY > upperBound[1]:
                    continue
                if min(Ay) > 0:
                    if maxAy < upperBound[0]:
                        upperBound[0] = maxAy
                        upperBound[1] = sumY
                    elif sumY < upperBound[1]:
                        upperBound[1] = sumY
                    ySet.append([list(y), maxAy, sumY])
                else:
                    for j in range(depth + 1, nr):
                        if Ay[j] == 0:
                            sear(j, y)
        tearUB = self.tear_upper_bound(G)
        A, _, cycleEdges = self.cycle_edge_matrix(G)
        nr, nc = A.shape
        if nr == 0:
            return [[[]], 0, 0]
        y_init = [False] * G.number_of_edges()
        for j in tearUB:
            y_init[j] = 1
        Ay_init = numpy.dot(A, y_init)
        upperBound = [max(Ay_init), sum(y_init)]
        y_init = [False] * G.number_of_edges()
        ySet = []
        sear(0, y_init)
        deleteSet = []
        for i in range(len(ySet)):
            if ySet[i][1] > upperBound[0]:
                deleteSet.append(i)
            elif ySet[i][1] == upperBound[0] and ySet[i][2] > upperBound[1]:
                deleteSet.append(i)
        for i in reversed(deleteSet):
            del ySet[i]
        deleteSet = []
        for i in range(len(ySet) - 1):
            if i in deleteSet:
                continue
            for j in range(i + 1, len(ySet)):
                if j in deleteSet:
                    continue
                for k in range(len(y_init)):
                    eq = True
                    if ySet[i][0][k] != ySet[j][0][k]:
                        eq = False
                        break
                if eq == True:
                    deleteSet.append(j)
        for i in reversed(sorted(deleteSet)):
            del ySet[i]
        es = []
        for y in ySet:
            edges = []
            for i in range(len(y[0])):
                if y[0][i] == 1:
                    edges.append(i)
            es.append(edges)
        return (es, upperBound[0], upperBound[1])

    def tear_upper_bound(self, G):
        """
        This function quickly finds a sub-optimal set of tear
        edges. This serves as an initial upperbound when looking
        for an optimal tear set. Having an initial upper bound
        improves efficiency.

        This works by constructing a search tree and just makes a
        tear set out of all the back edges.
        """

        def cyc(node, depth):
            depths[node] = depth
            depth += 1
            for edge in G.out_edges(node, keys=True):
                suc, key = (edge[1], edge[2])
                if depths[suc] is None:
                    parents[suc] = node
                    cyc(suc, depth)
                elif depths[suc] < depths[node]:
                    tearSet.append(edge_list.index((node, suc, key)))
        tearSet = []
        edge_list = self.idx_to_edge(G)
        depths = {}
        parents = {}
        for node in G.nodes:
            depths[node] = None
            parents[node] = None
        for node in G.nodes:
            if depths[node] is None:
                cyc(node, 0)
        return tearSet

    def sub_graph_edges(self, G, nodes):
        """
        This function returns a list of edge indexes that are
        included in a subgraph given by a list of nodes.

        Returns
        -------
            edges
                List of edge indexes in the subgraph
            inEdges
                List of edge indexes starting outside the subgraph
                and ending inside
            outEdges
                List of edge indexes starting inside the subgraph
                and ending outside
        """
        e = []
        ie = []
        oe = []
        edge_list = self.idx_to_edge(G)
        for i in range(G.number_of_edges()):
            src, dest, _ = edge_list[i]
            if src in nodes:
                if dest in nodes:
                    e.append(i)
                else:
                    oe.append(i)
            elif dest in nodes:
                ie.append(i)
        return (e, ie, oe)

    def cycle_edge_matrix(self, G):
        """
        Return a cycle-edge incidence matrix, a list of list of nodes in
        each cycle, and a list of list of edge indexes in each cycle.
        """
        cycleNodes, cycleEdges = self.all_cycles(G)
        ceMat = numpy.zeros((len(cycleEdges), G.number_of_edges()), dtype=numpy.dtype(int))
        for i in range(len(cycleEdges)):
            for e in cycleEdges[i]:
                ceMat[i, e] = 1
        return (ceMat, cycleNodes, cycleEdges)

    def all_cycles(self, G):
        """
        This function finds all the cycles in a directed graph.
        The algorithm is based on Tarjan 1973 Enumeration of the
        elementary circuits of a directed graph, SIAM J. Comput. v3 n2 1973.

        Returns
        -------
            cycleNodes
                List of lists of nodes in each cycle
            cycleEdges
                List of lists of edge indexes in each cycle
        """

        def backtrack(v, pre_key=None):
            f = False
            pointStack.append((v, pre_key))
            mark[v] = True
            markStack.append(v)
            sucs = list(adj[v])
            for si, key in sucs:
                if si < ni:
                    adj[v].remove((si, key))
                elif si == ni:
                    f = True
                    cyc = list(pointStack)
                    cyc.append((si, key))
                    cycles.append(cyc)
                elif not mark[si]:
                    g = backtrack(si, key)
                    f = f or g
            if f:
                while markStack[-1] != v:
                    u = markStack.pop()
                    mark[u] = False
                markStack.pop()
                mark[v] = False
            pointStack.pop()
            return f
        i2n, adj, _ = self.adj_lists(G, multi=True)
        pointStack = []
        markStack = []
        cycles = []
        mark = [False] * len(i2n)
        for ni in range(len(i2n)):
            backtrack(ni)
            while len(markStack) > 0:
                i = markStack.pop()
                mark[i] = False
        cycleNodes = []
        for cycle in cycles:
            cycleNodes.append([])
            for i in range(len(cycle)):
                ni, key = cycle[i]
                cycle[i] = (i2n[ni], key)
                cycleNodes[-1].append(i2n[ni])
            cycleNodes[-1].pop()
        edge_map = self.edge_to_idx(G)
        cycleEdges = []
        for cyc in cycles:
            ecyc = []
            for i in range(len(cyc) - 1):
                pre, suc, key = (cyc[i][0], cyc[i + 1][0], cyc[i + 1][1])
                ecyc.append(edge_map[pre, suc, key])
            cycleEdges.append(ecyc)
        return (cycleNodes, cycleEdges)

    def adj_lists(self, G, excludeEdges=None, nodes=None, multi=False):
        """
        Returns an adjacency list and a reverse adjacency list
        of node indexes for a MultiDiGraph.

        Arguments
        ---------
            G
                A networkx MultiDiGraph
            excludeEdges
                List of edge indexes to ignore when considering neighbors
            nodes
                List of nodes to form the adjacencies from
            multi
                If True, adjacency lists will contains tuples of
                (node, key) for every edge between two nodes

        Returns
        -------
            i2n
                Map from index to node for all nodes included in nodes
            adj
                Adjacency list of successor indexes
            adjR
                Reverse adjacency list of predecessor indexes
        """
        adj = []
        adjR = []
        exclude = set()
        if excludeEdges is not None:
            edge_list = self.idx_to_edge(G)
            for ei in excludeEdges:
                exclude.add(edge_list[ei])
        if nodes is None:
            nodes = self.idx_to_node(G)
        i2n = [None] * len(nodes)
        n2i = dict()
        i = -1
        for node in nodes:
            i += 1
            n2i[node] = i
            i2n[i] = node
        i = -1
        for node in nodes:
            i += 1
            adj.append([])
            adjR.append([])
            seen = set()
            for edge in G.out_edges(node, keys=True):
                suc, key = (edge[1], edge[2])
                if not multi and suc in seen:
                    continue
                if suc in nodes and edge not in exclude:
                    seen.add(suc)
                    if multi:
                        adj[i].append((n2i[suc], key))
                    else:
                        adj[i].append(n2i[suc])
            seen = set()
            for edge in G.in_edges(node, keys=True):
                pre, key = (edge[0], edge[2])
                if not multi and pre in seen:
                    continue
                if pre in nodes and edge not in exclude:
                    seen.add(pre)
                    if multi:
                        adjR[i].append((n2i[pre], key))
                    else:
                        adjR[i].append(n2i[pre])
        return (i2n, adj, adjR)