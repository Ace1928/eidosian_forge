from pyomo.core.base import Objective, Constraint
import array
def to_standard_form(self):
    """
    Produces a standard-form representation of the model. Returns
    the coefficient matrix (A), the cost vector (c), and the
    constraint vector (b), where the 'standard form' problem is

    min/max c'x
    s.t.    Ax = b
            x >= 0

    All three returned values are instances of the array.array
    class, and store Python floats (C doubles).
    """
    from pyomo.repn import generate_standard_repn
    colID = {}
    ID2name = {}
    id = 0
    tmp = self.variables().keys()
    tmp.sort()
    for v in tmp:
        colID[v] = id
        ID2name[id] = v
        id += 1
    var_id_map = {}
    leConstraints = {}
    geConstraints = {}
    eqConstraints = {}
    objectives = {}
    for c in self.component_map(active=True):
        if issubclass(c, Constraint):
            cons = self.component_map(c, active=True)
            for con_set_name in cons:
                con_set = cons[con_set_name]
                for ndx in con_set._data:
                    con = con_set._data[ndx]
                    terms = self._process_canonical_repn(generate_standard_repn(con.body, var_id_map))
                    if con.equality:
                        lb = self._process_canonical_repn(generate_standard_repn(con.lower, var_id_map))
                        for k in lb:
                            v = lb[k]
                            if k in terms:
                                terms[k] -= v
                            else:
                                terms[k] = -v
                        eqConstraints[con_set_name, ndx] = terms
                    else:
                        if con.upper is not None:
                            tmp = dict(terms)
                            ub = self._process_canonical_repn(generate_standard_repn(con.upper, var_id_map))
                            for k in ub:
                                if k in terms:
                                    tmp[k] -= ub[k]
                                else:
                                    tmp[k] = -ub[k]
                            leConstraints[con_set_name, ndx] = tmp
                        if con.lower is not None:
                            tmp = dict(terms)
                            lb = self._process_canonical_repn(generate_standard_repn(con.lower, var_id_map))
                            for k in lb:
                                if k in terms:
                                    tmp[k] -= lb[k]
                                else:
                                    tmp[k] = -lb[k]
                            geConstraints[con_set_name, ndx] = tmp
        elif issubclass(c, Objective):
            objs = self.component_map(c, active=True)
            for obj_set_name in objs:
                obj_set = objs[obj_set_name]
                for ndx in obj_set._data:
                    obj = obj_set._data[ndx]
                    terms = self._process_canonical_repn(generate_standard_repn(obj.expr, var_id_map))
                    objectives[obj_set_name, ndx] = terms
    nSlack = len(leConstraints)
    nExcess = len(geConstraints)
    nConstraints = len(leConstraints) + len(geConstraints) + len(eqConstraints)
    nVariables = len(colID) + nSlack + nExcess
    nRegVariables = len(colID)
    coefficients = array.array('d', [0] * nConstraints * nVariables)
    constraints = array.array('d', [0] * nConstraints)
    costs = array.array('d', [0] * nVariables)
    constraintID = 0
    for ndx in leConstraints:
        con = leConstraints[ndx]
        for termKey in con:
            coef = con[termKey]
            if termKey is None:
                constraints[constraintID] = -coef
            else:
                col = colID[termKey]
                coefficients[constraintID * nVariables + col] = coef
        coefficients[constraintID * nVariables + nRegVariables + constraintID] = 1
        constraintID += 1
    for ndx in geConstraints:
        con = geConstraints[ndx]
        for termKey in con:
            coef = con[termKey]
            if termKey is None:
                constraints[constraintID] = -coef
            else:
                col = colID[termKey]
                coefficients[constraintID * nVariables + col] = coef
        coefficients[constraintID * nVariables + nRegVariables + constraintID] = -1
        constraintID += 1
    for ndx in eqConstraints:
        con = eqConstraints[ndx]
        for termKey in con:
            coef = con[termKey]
            if termKey is None:
                constraints[constraintID] = -coef
            else:
                col = colID[termKey]
                coefficients[constraintID * nVariables + col] = coef
        constraintID += 1
    for obj_name in objectives:
        obj = objectives[obj_name]()
        for var in obj:
            costs[colID[var]] = obj[var]
    constraintPadding = 2
    numFmt = '% 1.4f'
    altFmt = '% 1.1g'
    maxColWidth = max(len(numFmt % 0.0), len(altFmt % 0.0))
    maxConstraintColWidth = max(len(numFmt % 0.0), len(altFmt % 0.0))
    maxConNameLen = 0
    conNames = []
    for name in leConstraints:
        strName = str(name)
        if len(strName) > maxConNameLen:
            maxConNameLen = len(strName)
        conNames.append(strName)
    for name in geConstraints:
        strName = str(name)
        if len(strName) > maxConNameLen:
            maxConNameLen = len(strName)
        conNames.append(strName)
    for name in eqConstraints:
        strName = str(name)
        if len(strName) > maxConNameLen:
            maxConNameLen = len(strName)
        conNames.append(strName)
    varNames = [None] * len(colID)
    for name in colID:
        tmp_name = ' ' + name
        if len(tmp_name) > maxColWidth:
            maxColWidth = len(tmp_name)
        varNames[colID[name]] = tmp_name
    for i in range(0, nSlack):
        tmp_name = ' _slack_%i' % i
        if len(tmp_name) > maxColWidth:
            maxColWidth = len(tmp_name)
        varNames.append(tmp_name)
    for i in range(0, nExcess):
        tmp_name = ' _excess_%i' % i
        if len(tmp_name) > maxColWidth:
            maxColWidth = len(tmp_name)
        varNames.append(tmp_name)
    line = ' ' * maxConNameLen + ' ' * constraintPadding + ' '
    for col in range(0, nVariables):
        token = varNames[col]
        token += ' ' * (maxColWidth - len(token))
        line += ' ' + token + ' '
    print(line + '\n')
    print(' ' * maxConNameLen + ' ' * constraintPadding + '+--' + ' ' * ((maxColWidth + 2) * nVariables - 4) + '--+' + '\n')
    line = ' ' * maxConNameLen + ' ' * constraintPadding + '|'
    for col in range(0, nVariables):
        token = numFmt % costs[col]
        if len(token) > maxColWidth:
            token = altFmt % costs[col]
        token += ' ' * (maxColWidth - len(token))
        line += ' ' + token + ' '
    line += '|'
    print(line + '\n')
    print(' ' * maxConNameLen + ' ' * constraintPadding + '+--' + ' ' * ((maxColWidth + 2) * nVariables - 4) + '--+' + '\n')
    print(' ' * maxConNameLen + ' ' * constraintPadding + '+--' + ' ' * ((maxColWidth + 2) * nVariables - 4) + '--+' + ' ' * constraintPadding + '+--' + ' ' * (maxConstraintColWidth - 1) + '--+' + '\n')
    for row in range(0, nConstraints):
        line = conNames[row] + ' ' * constraintPadding + ' ' * (maxConNameLen - len(conNames[row])) + '|'
        for col in range(0, nVariables):
            token = numFmt % coefficients[nVariables * row + col]
            if len(token) > maxColWidth:
                token = altFmt % coefficients[nVariables * row + col]
            token += ' ' * (maxColWidth - len(token))
            line += ' ' + token + ' '
        line += '|' + ' ' * constraintPadding + '|'
        token = numFmt % constraints[row]
        if len(token) > maxConstraintColWidth:
            token = altFmt % constraints[row]
        token += ' ' * (maxConstraintColWidth - len(token))
        line += ' ' + token + '  |'
        print(line + '\n')
    print(' ' * maxConNameLen + ' ' * constraintPadding + '+--' + ' ' * ((maxColWidth + 2) * nVariables - 4) + '--+' + ' ' * constraintPadding + '+--' + ' ' * (maxConstraintColWidth - 1) + '--+' + '\n')
    return (coefficients, costs, constraints)