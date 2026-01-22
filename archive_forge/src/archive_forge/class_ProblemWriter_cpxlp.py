import logging
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
@WriterFactory.register('cpxlp_v1', 'Generate the corresponding CPLEX LP file')
@WriterFactory.register('lp_v1', 'Generate the corresponding CPLEX LP file')
class ProblemWriter_cpxlp(AbstractProblemWriter):

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.cpxlp)
        self._referenced_variable_ids = {}
        self._precision_string = '.17g'
        self.linear_coef_string_template = '%+' + self._precision_string + ' %s\n'
        self.quad_coef_string_template = '%+' + self._precision_string + ' '
        self.obj_string_template = '%+' + self._precision_string + ' %s\n'
        self.sos_template_string = '%s:%' + self._precision_string + '\n'
        self.eq_string_template = '= %' + self._precision_string + '\n'
        self.geq_string_template = '>= %' + self._precision_string + '\n\n'
        self.leq_string_template = '<= %' + self._precision_string + '\n\n'
        self.lb_string_template = '%' + self._precision_string + ' <= '
        self.ub_string_template = ' <= %' + self._precision_string + '\n'

    def __call__(self, model, output_filename, solver_capability, io_options):
        io_options = dict(io_options)
        skip_trivial_constraints = io_options.pop('skip_trivial_constraints', False)
        symbolic_solver_labels = io_options.pop('symbolic_solver_labels', False)
        output_fixed_variable_bounds = io_options.pop('output_fixed_variable_bounds', False)
        include_all_variable_bounds = io_options.pop('include_all_variable_bounds', False)
        labeler = io_options.pop('labeler', None)
        file_determinism = io_options.pop('file_determinism', 1)
        row_order = io_options.pop('row_order', None)
        column_order = io_options.pop('column_order', None)
        force_objective_constant = io_options.pop('force_objective_constant', False)
        if len(io_options):
            raise ValueError('ProblemWriter_cpxlp passed unrecognized io_options:\n\t' + '\n\t'.join(('%s = %s' % (k, v) for k, v in io_options.items())))
        if symbolic_solver_labels and labeler is not None:
            raise ValueError("ProblemWriter_cpxlp: Using both the 'symbolic_solver_labels' and 'labeler' I/O options is forbidden")
        if symbolic_solver_labels:
            labeler = TextLabeler()
        elif labeler is None:
            labeler = NumericLabeler('x')
        self._referenced_variable_ids.clear()
        if output_filename is None:
            output_filename = model.name + '.lp'
        with PauseGC() as pgc:
            with open(output_filename, 'w') as output_file:
                symbol_map = self._print_model_LP(model, output_file, solver_capability, labeler, output_fixed_variable_bounds=output_fixed_variable_bounds, file_determinism=file_determinism, row_order=row_order, column_order=column_order, skip_trivial_constraints=skip_trivial_constraints, force_objective_constant=force_objective_constant, include_all_variable_bounds=include_all_variable_bounds)
        self._referenced_variable_ids.clear()
        return (output_filename, symbol_map)

    def _print_expr_canonical(self, x, output, object_symbol_dictionary, variable_symbol_dictionary, is_objective, column_order, file_determinism, force_objective_constant=False):
        """
        Return a expression as a string in LP format.

        Note that this function does not handle any differences in LP format
        interpretation by the solvers (e.g. CPlex vs GLPK).  That decision is
        left up to the caller.

        required arguments:
          x: A Pyomo canonical expression to write in LP format
        """
        assert not force_objective_constant or is_objective
        linear_coef_string_template = self.linear_coef_string_template
        quad_coef_string_template = self.quad_coef_string_template
        constant = True
        if len(x.linear_vars) > 0:
            constant = False
            for vardata in x.linear_vars:
                self._referenced_variable_ids[id(vardata)] = vardata
            if column_order is None:
                names = [variable_symbol_dictionary[id(var)] for var in x.linear_vars]
                term_iterator = zip(x.linear_coefs, names)
                if file_determinism > 0:
                    term_iterator = sorted(term_iterator, key=lambda x: x[1])
                for coef, name in term_iterator:
                    output.append(linear_coef_string_template % (coef, name))
            else:
                for i, var in sorted(enumerate(x.linear_vars), key=lambda x: column_order[x[1]]):
                    name = variable_symbol_dictionary[id(var)]
                    output.append(linear_coef_string_template % (x.linear_coefs[i], name))
        if len(x.quadratic_vars) > 0:
            constant = False
            for var1, var2 in x.quadratic_vars:
                self._referenced_variable_ids[id(var1)] = var1
                self._referenced_variable_ids[id(var2)] = var2
            output.append('+ [\n')
            if column_order is None:
                quad = set()
                names = []
                i = 0
                for var1, var2 in x.quadratic_vars:
                    name1 = variable_symbol_dictionary[id(var1)]
                    name2 = variable_symbol_dictionary[id(var2)]
                    if name1 < name2:
                        names.append((name1, name2))
                    elif name1 > name2:
                        names.append((name2, name1))
                    else:
                        quad.add(i)
                        names.append((name1, name1))
                    i += 1
                term_iterator = enumerate(names)
                if file_determinism > 0:
                    term_iterator = sorted(term_iterator, key=lambda x: x[1])
                for i, names_ in term_iterator:
                    if is_objective:
                        tmp = 2 * x.quadratic_coefs[i]
                        output.append(quad_coef_string_template % tmp)
                    else:
                        output.append(quad_coef_string_template % x.quadratic_coefs[i])
                    if i in quad:
                        output.append('%s ^ 2\n' % names_[0])
                    else:
                        output.append('%s * %s\n' % (names_[0], names_[1]))
            else:
                quad = set()
                cols = []
                i = 0
                for var1, var2 in x.quadratic_vars:
                    col1 = column_order[var1]
                    col2 = column_order[var2]
                    if col1 < col2:
                        cols.append(((col1, col2), variable_symbol_dictionary[id(var1)], variable_symbol_dictionary[id(var2)]))
                    elif col1 > col2:
                        cols.append(((col2, col1), variable_symbol_dictionary[id(var2)], variable_symbol_dictionary[id(var1)]))
                    else:
                        quad.add(i)
                        cols.append(((col1, col1), variable_symbol_dictionary[id(var1)]))
                    i += 1
                for i, cols_ in sorted(enumerate(cols), key=lambda x: x[1][0]):
                    if is_objective:
                        output.append(quad_coef_string_template % 2 * x.quadratic_coefs[i])
                    else:
                        output.append(quad_coef_string_template % x.quadratic_coefs[i])
                    if i in quad:
                        output.append('%s ^ 2\n' % cols_[1])
                    else:
                        output.append('%s * %s\n' % (cols_[1], cols_[2]))
            output.append(']')
            if is_objective:
                output.append(' / 2\n')
            else:
                output.append('\n')
        if constant and (not is_objective):
            output.append(linear_coef_string_template % (0, 'ONE_VAR_CONSTANT'))
        if is_objective and (force_objective_constant or x.constant != 0.0):
            output.append(self.obj_string_template % (x.constant, 'ONE_VAR_CONSTANT'))
        return x.constant

    def printSOS(self, symbol_map, labeler, variable_symbol_map, soscondata, output):
        """
        Prints the SOS constraint associated with the _SOSConstraintData object
        """
        sos_template_string = self.sos_template_string
        if hasattr(soscondata, 'get_items'):
            sos_items = list(soscondata.get_items())
        else:
            sos_items = list(soscondata.items())
        if len(sos_items) == 0:
            return
        level = soscondata.level
        output.append('%s: S%s::\n' % (symbol_map.getSymbol(soscondata, labeler), level))
        for vardata, weight in sos_items:
            weight = _get_bound(weight)
            if weight < 0:
                raise ValueError('Cannot use negative weight %f for variable %s is special ordered set %s ' % (weight, vardata.name, soscondata.name))
            if vardata.fixed:
                raise RuntimeError("SOSConstraint '%s' includes a fixed variable '%s'. This is currently not supported. Deactivate this constraint in order to proceed." % (soscondata.name, vardata.name))
            self._referenced_variable_ids[id(vardata)] = vardata
            output.append(sos_template_string % (variable_symbol_map.getSymbol(vardata), weight))

    def _print_model_LP(self, model, output_file, solver_capability, labeler, output_fixed_variable_bounds=False, file_determinism=1, row_order=None, column_order=None, skip_trivial_constraints=False, force_objective_constant=False, include_all_variable_bounds=False):
        eq_string_template = self.eq_string_template
        leq_string_template = self.leq_string_template
        geq_string_template = self.geq_string_template
        ub_string_template = self.ub_string_template
        lb_string_template = self.lb_string_template
        symbol_map = SymbolMap()
        variable_symbol_map = SymbolMap()
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias
        variable_label_pairs = []
        sortOrder = SortComponents.unsorted
        if file_determinism >= 1:
            sortOrder = sortOrder | SortComponents.indices
            if file_determinism >= 2:
                sortOrder = sortOrder | SortComponents.alphabetical
        all_blocks = []
        variable_list = []
        all_blocks = list(model.block_data_objects(active=True, sort=sortOrder))
        variable_list = list(model.component_data_objects(Var, sort=sortOrder))
        variable_label_pairs = list(((vardata, create_symbol_func(symbol_map, vardata, labeler)) for vardata in variable_list))
        variable_symbol_map.addSymbols(variable_label_pairs)
        object_symbol_dictionary = symbol_map.byObject
        variable_symbol_dictionary = variable_symbol_map.byObject

        def print_expr_canonical(obj, x, output, object_symbol_dictionary, variable_symbol_dictionary, is_objective, column_order, file_determinism, force_objective_constant=False):
            try:
                return self._print_expr_canonical(x=x, output=output, object_symbol_dictionary=object_symbol_dictionary, variable_symbol_dictionary=variable_symbol_dictionary, is_objective=is_objective, column_order=column_order, file_determinism=file_determinism, force_objective_constant=force_objective_constant)
            except KeyError as e:
                _id = e.args[0]
                _var = None
                if x.linear_vars:
                    for v in x.linear_vars:
                        if id(v) == _id:
                            _var = v
                            break
                if _var is None and x.quadratic_vars:
                    for v in x.quadratic_vars:
                        v = [_v for _v in v if id(_v) == _id]
                        if v:
                            _var = v[0]
                            break
                if _var is not None:
                    logger.error('Model contains an expression (%s) that contains a variable (%s) that is not attached to an active block on the submodel being written' % (obj.name, _var.name))
                raise
        output = []
        output.append('\\* Source Pyomo model name=%s *\\\n\n' % (model.name,))
        supports_quadratic_objective = solver_capability('quadratic_objective')
        numObj = 0
        onames = []
        for block in all_blocks:
            gen_obj_repn = getattr(block, '_gen_obj_repn', None)
            if gen_obj_repn is not None:
                gen_obj_repn = bool(gen_obj_repn)
                if not hasattr(block, '_repn'):
                    block._repn = ComponentMap()
                block_repn = block._repn
            for objective_data in block.component_data_objects(Objective, active=True, sort=sortOrder, descend_into=False):
                numObj += 1
                onames.append(objective_data.name)
                if numObj > 1:
                    raise ValueError("More than one active objective defined for input model '%s'; Cannot write legal LP file\nObjectives: %s" % (model.name, ' '.join(onames)))
                create_symbol_func(symbol_map, objective_data, labeler)
                symbol_map.alias(objective_data, '__default_objective__')
                if objective_data.is_minimizing():
                    output.append('min \n')
                else:
                    output.append('max \n')
                if gen_obj_repn == False:
                    repn = block_repn[objective_data]
                else:
                    repn = generate_standard_repn(objective_data.expr, quadratic=supports_quadratic_objective)
                    if gen_obj_repn:
                        block_repn[objective_data] = repn
                degree = repn.polynomial_degree()
                if degree == 0:
                    logger.warning('Constant objective detected, replacing with a placeholder to prevent solver failure.')
                    force_objective_constant = True
                elif degree == 2:
                    if not supports_quadratic_objective:
                        raise RuntimeError('Selected solver is unable to handle objective functions with quadratic terms. Objective at issue: %s.' % objective_data.name)
                elif degree is None:
                    raise RuntimeError("Cannot write legal LP file.  Objective '%s' has nonlinear terms that are not quadratic." % objective_data.name)
                output.append(object_symbol_dictionary[id(objective_data)] + ':\n')
                offset = print_expr_canonical(objective_data, repn, output, object_symbol_dictionary, variable_symbol_dictionary, True, column_order, file_determinism, force_objective_constant=force_objective_constant)
        if numObj == 0:
            raise ValueError('ERROR: No objectives defined for input model. Cannot write legal LP file.')
        output.append('\n')
        output.append('s.t.\n')
        output.append('\n')
        have_nontrivial = False
        supports_quadratic_constraint = solver_capability('quadratic_constraint')

        def constraint_generator():
            for block in all_blocks:
                gen_con_repn = getattr(block, '_gen_con_repn', None)
                if gen_con_repn is not None:
                    gen_con_repn = bool(gen_con_repn)
                    if not hasattr(block, '_repn'):
                        block._repn = ComponentMap()
                    block_repn = block._repn
                for constraint_data in block.component_data_objects(Constraint, active=True, sort=sortOrder, descend_into=False):
                    if not constraint_data.has_lb() and (not constraint_data.has_ub()):
                        assert not constraint_data.equality
                        continue
                    if gen_con_repn == False:
                        repn = block_repn[constraint_data]
                    else:
                        if constraint_data._linear_canonical_form:
                            repn = constraint_data.canonical_form()
                        else:
                            repn = generate_standard_repn(constraint_data.body, quadratic=supports_quadratic_constraint)
                        if gen_con_repn:
                            block_repn[constraint_data] = repn
                    yield (constraint_data, repn)
        if row_order is not None:
            sorted_constraint_list = list(constraint_generator())
            sorted_constraint_list.sort(key=lambda x: row_order[x[0]])

            def yield_all_constraints():
                yield from sorted_constraint_list
        else:
            yield_all_constraints = constraint_generator
        for constraint_data, repn in yield_all_constraints():
            have_nontrivial = True
            degree = repn.polynomial_degree()
            if degree == 0:
                if skip_trivial_constraints:
                    continue
            elif degree == 2:
                if not supports_quadratic_constraint:
                    raise ValueError("Solver unable to handle quadratic expressions. Constraint at issue: '%s'" % constraint_data.name)
            elif degree is None:
                raise ValueError("Cannot write legal LP file.  Constraint '%s' has a body with nonlinear terms." % constraint_data.name)
            con_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
            if constraint_data.equality:
                assert value(constraint_data.lower) == value(constraint_data.upper)
                label = 'c_e_%s_' % con_symbol
                alias_symbol_func(symbol_map, constraint_data, label)
                output.append(label)
                output.append(':\n')
                offset = print_expr_canonical(constraint_data, repn, output, object_symbol_dictionary, variable_symbol_dictionary, False, column_order, file_determinism)
                bound = constraint_data.lower
                bound = _get_bound(bound) - offset
                output.append(eq_string_template % _no_negative_zero(bound))
                output.append('\n')
            else:
                if constraint_data.has_lb():
                    if constraint_data.has_ub():
                        label = 'r_l_%s_' % con_symbol
                    else:
                        label = 'c_l_%s_' % con_symbol
                    alias_symbol_func(symbol_map, constraint_data, label)
                    output.append(label)
                    output.append(':\n')
                    offset = print_expr_canonical(constraint_data, repn, output, object_symbol_dictionary, variable_symbol_dictionary, False, column_order, file_determinism)
                    bound = constraint_data.lower
                    bound = _get_bound(bound) - offset
                    output.append(geq_string_template % _no_negative_zero(bound))
                else:
                    assert constraint_data.has_ub()
                if constraint_data.has_ub():
                    if constraint_data.has_lb():
                        label = 'r_u_%s_' % con_symbol
                    else:
                        label = 'c_u_%s_' % con_symbol
                    alias_symbol_func(symbol_map, constraint_data, label)
                    output.append(label)
                    output.append(':\n')
                    offset = print_expr_canonical(constraint_data, repn, output, object_symbol_dictionary, variable_symbol_dictionary, False, column_order, file_determinism)
                    bound = constraint_data.upper
                    bound = _get_bound(bound) - offset
                    output.append(leq_string_template % _no_negative_zero(bound))
                else:
                    assert constraint_data.has_lb()
            if len(output) > 1024:
                output_file.write(''.join(output))
                output = []
        if not have_nontrivial:
            logger.warning('Empty constraint block written in LP format - solver may error')
        prefix = ''
        output.append('%sc_e_ONE_VAR_CONSTANT: \n' % prefix)
        output.append('%sONE_VAR_CONSTANT = 1.0\n' % prefix)
        output.append('\n')
        SOSlines = []
        sos1 = solver_capability('sos1')
        sos2 = solver_capability('sos2')
        writtenSOS = False
        for block in all_blocks:
            for soscondata in block.component_data_objects(SOSConstraint, active=True, sort=sortOrder, descend_into=False):
                create_symbol_func(symbol_map, soscondata, labeler)
                level = soscondata.level
                if level == 1 and (not sos1) or (level == 2 and (not sos2)) or level > 2:
                    raise ValueError('Solver does not support SOS level %s constraints' % level)
                if writtenSOS == False:
                    SOSlines.append('SOS\n')
                    writtenSOS = True
                self.printSOS(symbol_map, labeler, variable_symbol_map, soscondata, SOSlines)
        output.append('bounds\n')
        integer_vars = []
        binary_vars = []
        for vardata in variable_list:
            if not include_all_variable_bounds and id(vardata) not in self._referenced_variable_ids:
                continue
            name_to_output = variable_symbol_dictionary[id(vardata)]
            if name_to_output == 'e':
                raise ValueError("Attempting to write variable with name 'e' in a CPLEX LP formatted file will cause a parse failure due to confusion with numeric values expressed in scientific notation")
            if vardata.is_binary():
                binary_vars.append(name_to_output)
            elif vardata.is_integer():
                integer_vars.append(name_to_output)
            elif not vardata.is_continuous():
                raise TypeError("Invalid domain type for variable with name '%s'. Variable is not continuous, integer, or binary." % vardata.name)
            if vardata.fixed:
                if not output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective or constraint expression on model %s, which is usually indicative of a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' to suppress this error and fix the variable by overwriting its bounds in the LP file." % (vardata.name, model.name))
                if vardata.value is None:
                    raise ValueError('Variable cannot be fixed to a value of None.')
                vardata_lb = value(vardata.value)
                vardata_ub = value(vardata.value)
                output.append('   ')
                output.append(lb_string_template % _no_negative_zero(vardata_lb))
                output.append(name_to_output)
                output.append(ub_string_template % _no_negative_zero(vardata_ub))
            else:
                vardata_lb = _get_bound(vardata.lb)
                vardata_ub = _get_bound(vardata.ub)
                output.append('   ')
                if vardata.has_lb():
                    output.append(lb_string_template % _no_negative_zero(vardata_lb))
                else:
                    output.append(' -inf <= ')
                output.append(name_to_output)
                if vardata.has_ub():
                    output.append(ub_string_template % _no_negative_zero(vardata_ub))
                else:
                    output.append(' <= +inf\n')
        if len(integer_vars) > 0:
            output.append('general\n')
            for var_name in integer_vars:
                output.append('  %s\n' % var_name)
        if len(binary_vars) > 0:
            output.append('binary\n')
            for var_name in binary_vars:
                output.append('  %s\n' % var_name)
        output.append(''.join(SOSlines))
        output.append('end\n')
        output_file.write(''.join(output))
        vars_to_delete = set(variable_symbol_map.byObject.keys()) - set(self._referenced_variable_ids.keys())
        sm_byObject = symbol_map.byObject
        sm_bySymbol = symbol_map.bySymbol
        var_sm_byObject = variable_symbol_map.byObject
        for varid in vars_to_delete:
            symbol = var_sm_byObject[varid]
            del sm_byObject[varid]
            del sm_bySymbol[symbol]
        del variable_symbol_map
        return symbol_map