from io import StringIO
from pyomo.common.gc_manager import PauseGC
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
import logging
@WriterFactory.register('gams', 'Generate the corresponding GAMS file')
class ProblemWriter_gams(AbstractProblemWriter):

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.gams)

    def __call__(self, model, output_filename, solver_capability, io_options):
        """
        Write a model in the GAMS modeling language format.

        Keyword Arguments
        -----------------
        output_filename: str
            Name of file to write GAMS model to. Optionally pass a file-like
            stream and the model will be written to that instead.
        io_options: dict
            - warmstart=True
                Warmstart by initializing model's variables to their values.
            - symbolic_solver_labels=False
                Use full Pyomo component names rather than
                shortened symbols (slower, but useful for debugging).
            - labeler=None
                Custom labeler. Incompatible with symbolic_solver_labels.
            - solver=None
                If None, GAMS will use default solver for model type.
            - mtype=None
                Model type. If None, will chose from lp, nlp, mip, and minlp.
            - add_options=None
                List of additional lines to write directly
                into model file before the solve statement.
                For model attributes, <model name> is GAMS_MODEL.
            - skip_trivial_constraints=False
                Skip writing constraints whose body section is fixed.
            - output_fixed_variables=False
                If True, output fixed variables as variables; otherwise,
                output numeric value.
            - file_determinism=1
                | How much effort do we want to put into ensuring the
                | GAMS file is written deterministically for a Pyomo model:
                |     0 : None
                |     1 : sort keys of indexed components (default)
                |     2 : sort keys AND sort names (over declaration order)
            - put_results=None
                Filename for optionally writing solution values and
                marginals.  If put_results_format is 'gdx', then GAMS
                will write solution values and marginals to
                GAMS_MODEL_p.gdx and solver statuses to
                {put_results}_s.gdx.  If put_results_format is 'dat',
                then solution values and marginals are written to
                (put_results).dat, and solver statuses to (put_results +
                'stat').dat.
            - put_results_format='gdx'
                Format used for put_results, one of 'gdx', 'dat'.

        """
        io_options = dict(io_options)
        symbolic_solver_labels = io_options.pop('symbolic_solver_labels', False)
        labeler = io_options.pop('labeler', None)
        solver = io_options.pop('solver', None)
        mtype = io_options.pop('mtype', None)
        solprint = io_options.pop('solprint', 'off')
        limrow = io_options.pop('limrow', 0)
        limcol = io_options.pop('limcol', 0)
        solvelink = io_options.pop('solvelink', 5)
        add_options = io_options.pop('add_options', None)
        skip_trivial_constraints = io_options.pop('skip_trivial_constraints', False)
        output_fixed_variables = io_options.pop('output_fixed_variables', False)
        file_determinism = io_options.pop('file_determinism', 1)
        sorter_map = {0: SortComponents.unsorted, 1: SortComponents.deterministic, 2: SortComponents.sortBoth}
        sort = sorter_map[file_determinism]
        warmstart = io_options.pop('warmstart', True)
        put_results = io_options.pop('put_results', None)
        put_results_format = io_options.pop('put_results_format', 'gdx')
        assert put_results_format in ('gdx', 'dat')
        if len(io_options):
            raise ValueError('GAMS writer passed unrecognized io_options:\n\t' + '\n\t'.join(('%s = %s' % (k, v) for k, v in io_options.items())))
        if solver is not None and solver.upper() not in valid_solvers:
            raise ValueError('GAMS writer passed unrecognized solver: %s' % solver)
        if mtype is not None:
            valid_mtypes = set(['lp', 'qcp', 'nlp', 'dnlp', 'rmip', 'mip', 'rmiqcp', 'rminlp', 'miqcp', 'minlp', 'rmpec', 'mpec', 'mcp', 'cns', 'emp'])
            if mtype.lower() not in valid_mtypes:
                raise ValueError('GAMS writer passed unrecognized model type: %s' % mtype)
            if solver is not None and mtype.upper() not in valid_solvers[solver.upper()]:
                raise ValueError('GAMS writer passed solver (%s) unsuitable for given model type (%s)' % (solver, mtype))
        if output_filename is None:
            output_filename = model.name + '.gms'
        if symbolic_solver_labels and labeler is not None:
            raise ValueError("GAMS writer: Using both the 'symbolic_solver_labels' and 'labeler' I/O options is forbidden")
        if symbolic_solver_labels:
            var_labeler = con_labeler = ShortNameLabeler(60, prefix='s_', suffix='_', caseInsensitive=True, legalRegex='^[a-zA-Z]')
        elif labeler is None:
            var_labeler = NumericLabeler('x')
            con_labeler = NumericLabeler('c')
        else:
            var_labeler = con_labeler = labeler
        var_list = []
        symbolMap = GAMSSymbolMap(var_labeler, var_list)
        with PauseGC() as pgc:
            try:
                if isinstance(output_filename, str):
                    output_file = open(output_filename, 'w')
                else:
                    output_file = output_filename
                self._write_model(model=model, output_file=output_file, solver_capability=solver_capability, var_list=var_list, var_label=symbolMap.var_label, symbolMap=symbolMap, con_labeler=con_labeler, sort=sort, skip_trivial_constraints=skip_trivial_constraints, output_fixed_variables=output_fixed_variables, warmstart=warmstart, solver=solver, mtype=mtype, solprint=solprint, limrow=limrow, limcol=limcol, solvelink=solvelink, add_options=add_options, put_results=put_results, put_results_format=put_results_format)
            finally:
                if isinstance(output_filename, str):
                    output_file.close()
        return (output_filename, symbolMap)

    def _write_model(self, model, output_file, solver_capability, var_list, var_label, symbolMap, con_labeler, sort, skip_trivial_constraints, output_fixed_variables, warmstart, solver, mtype, solprint, limrow, limcol, solvelink, add_options, put_results, put_results_format):
        constraint_names = []
        ConstraintIO = StringIO()
        linear = True
        linear_degree = set([0, 1])
        dnlp = False
        model_ctypes = model.collect_ctypes(active=True)
        invalids = set()
        for t in model_ctypes - valid_active_ctypes_minlp:
            if issubclass(t, ActiveComponent):
                invalids.add(t)
        if len(invalids):
            invalids = [t.__name__ for t in invalids]
            raise RuntimeError('Unallowable active component(s) %s.\nThe GAMS writer cannot export models with this component type.' % ', '.join(invalids))
        tc = StorageTreeChecker(model)
        for con in model.component_data_objects(Constraint, active=True, sort=sort):
            if not con.has_lb() and (not con.has_ub()):
                assert not con.equality
                continue
            con_body = as_numeric(con.body)
            if skip_trivial_constraints and con_body.is_fixed():
                continue
            if linear:
                if con_body.polynomial_degree() not in linear_degree:
                    linear = False
            cName = symbolMap.getSymbol(con, con_labeler)
            con_body_str, con_discontinuous = expression_to_string(con_body, tc, smap=symbolMap, output_fixed_variables=output_fixed_variables)
            dnlp |= con_discontinuous
            if con.equality:
                constraint_names.append('%s' % cName)
                ConstraintIO.write('%s.. %s =e= %s ;\n' % (constraint_names[-1], con_body_str, ftoa(con.upper, False)))
            else:
                if con.has_lb():
                    constraint_names.append('%s_lo' % cName)
                    ConstraintIO.write('%s.. %s =l= %s ;\n' % (constraint_names[-1], ftoa(con.lower, False), con_body_str))
                if con.has_ub():
                    constraint_names.append('%s_hi' % cName)
                    ConstraintIO.write('%s.. %s =l= %s ;\n' % (constraint_names[-1], con_body_str, ftoa(con.upper, False)))
        obj = list(model.component_data_objects(Objective, active=True, sort=sort))
        if len(obj) != 1:
            raise RuntimeError('GAMS writer requires exactly one active objective (found %s)' % len(obj))
        obj = obj[0]
        if linear:
            if obj.polynomial_degree() not in linear_degree:
                linear = False
        obj_expr_str, obj_discontinuous = expression_to_string(obj.expr, tc, smap=symbolMap, output_fixed_variables=output_fixed_variables)
        dnlp |= obj_discontinuous
        oName = symbolMap.getSymbol(obj, con_labeler)
        constraint_names.append(oName)
        ConstraintIO.write('%s.. GAMS_OBJECTIVE =e= %s ;\n' % (oName, obj_expr_str))
        categorized_vars = Categorizer(var_list, symbolMap)
        output_file.write('$offlisting\n')
        output_file.write('$offdigit\n\n')
        output_file.write('EQUATIONS\n\t')
        output_file.write('\n\t'.join(constraint_names))
        if categorized_vars.binary:
            output_file.write(';\n\nBINARY VARIABLES\n\t')
            output_file.write('\n\t'.join(categorized_vars.binary))
        if categorized_vars.ints:
            output_file.write(';\n\nINTEGER VARIABLES')
            output_file.write('\n\t')
            output_file.write('\n\t'.join(categorized_vars.ints))
        if categorized_vars.positive:
            output_file.write(';\n\nPOSITIVE VARIABLES\n\t')
            output_file.write('\n\t'.join(categorized_vars.positive))
        output_file.write(';\n\nVARIABLES\n\tGAMS_OBJECTIVE\n\t')
        output_file.write('\n\t'.join(categorized_vars.reals + categorized_vars.fixed))
        output_file.write(';\n\n')
        for var in categorized_vars.fixed:
            output_file.write('%s.fx = %s;\n' % (var, ftoa(value(symbolMap.getObject(var)), False)))
        output_file.write('\n')
        for line in ConstraintIO.getvalue().splitlines():
            if len(line) > 80000:
                line = split_long_line(line)
            output_file.write(line + '\n')
        output_file.write('\n')
        warn_int_bounds = False
        for category, var_name in categorized_vars:
            var = symbolMap.getObject(var_name)
            tc(var)
            lb, ub = var.bounds
            if category == 'positive':
                if ub is not None:
                    output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
            elif category == 'ints':
                if lb is None:
                    warn_int_bounds = True
                    logger.warning('Lower bound for integer variable %s set to -1.0E+100.' % var.name)
                    output_file.write('%s.lo = -1.0E+100;\n' % var_name)
                elif lb != 0:
                    output_file.write('%s.lo = %s;\n' % (var_name, ftoa(lb, False)))
                if ub is None:
                    warn_int_bounds = True
                    logger.warning('Upper bound for integer variable %s set to +1.0E+100.' % var.name)
                    output_file.write('%s.up = +1.0E+100;\n' % var_name)
                else:
                    output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
            elif category == 'binary':
                if lb != 0:
                    output_file.write('%s.lo = %s;\n' % (var_name, ftoa(lb, False)))
                if ub != 1:
                    output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
            elif category == 'reals':
                if lb is not None:
                    output_file.write('%s.lo = %s;\n' % (var_name, ftoa(lb, False)))
                if ub is not None:
                    output_file.write('%s.up = %s;\n' % (var_name, ftoa(ub, False)))
            else:
                raise KeyError('Category %s not supported' % category)
            if warmstart and var.value is not None:
                output_file.write('%s.l = %s;\n' % (var_name, ftoa(var.value, False)))
        if warn_int_bounds:
            logger.warning('GAMS requires finite bounds for integer variables. 1.0E100 is as extreme as GAMS will define, and should be enough to appear unbounded. If the solver cannot handle this bound, explicitly set a smaller bound on the pyomo model, or try a different GAMS solver.')
        model_name = 'GAMS_MODEL'
        output_file.write('\nMODEL %s /all/ ;\n' % model_name)
        if mtype is None:
            mtype = ('lp', 'nlp', 'mip', 'minlp')[(0 if linear else 1) + (2 if categorized_vars.binary or categorized_vars.ints else 0)]
            if mtype == 'nlp' and dnlp:
                mtype = 'dnlp'
        if solver is not None:
            if mtype.upper() not in valid_solvers[solver.upper()]:
                raise ValueError('GAMS writer passed solver (%s) unsuitable for model type (%s)' % (solver, mtype))
            output_file.write('option %s=%s;\n' % (mtype, solver))
        output_file.write('option solprint=%s;\n' % solprint)
        output_file.write('option limrow=%d;\n' % limrow)
        output_file.write('option limcol=%d;\n' % limcol)
        output_file.write('option solvelink=%d;\n' % solvelink)
        if put_results is not None and put_results_format == 'gdx':
            output_file.write('option savepoint=1;\n')
        if add_options is not None:
            output_file.write('\n* START USER ADDITIONAL OPTIONS\n')
            for line in add_options:
                output_file.write('\n' + line)
            output_file.write('\n\n* END USER ADDITIONAL OPTIONS\n\n')
        output_file.write('SOLVE %s USING %s %simizing GAMS_OBJECTIVE;\n\n' % (model_name, mtype, 'min' if obj.sense == minimize else 'max'))
        stat_vars = ['MODELSTAT', 'SOLVESTAT', 'OBJEST', 'OBJVAL', 'NUMVAR', 'NUMEQU', 'NUMDVAR', 'NUMNZ', 'ETSOLVE']
        output_file.write("Scalars MODELSTAT 'model status', SOLVESTAT 'solve status';\n")
        output_file.write('MODELSTAT = %s.modelstat;\n' % model_name)
        output_file.write('SOLVESTAT = %s.solvestat;\n\n' % model_name)
        output_file.write("Scalar OBJEST 'best objective', OBJVAL 'objective value';\n")
        output_file.write('OBJEST = %s.objest;\n' % model_name)
        output_file.write('OBJVAL = %s.objval;\n\n' % model_name)
        output_file.write("Scalar NUMVAR 'number of variables';\n")
        output_file.write('NUMVAR = %s.numvar\n\n' % model_name)
        output_file.write("Scalar NUMEQU 'number of equations';\n")
        output_file.write('NUMEQU = %s.numequ\n\n' % model_name)
        output_file.write("Scalar NUMDVAR 'number of discrete variables';\n")
        output_file.write('NUMDVAR = %s.numdvar\n\n' % model_name)
        output_file.write("Scalar NUMNZ 'number of nonzeros';\n")
        output_file.write('NUMNZ = %s.numnz\n\n' % model_name)
        output_file.write("Scalar ETSOLVE 'time to execute solve statement';\n")
        output_file.write('ETSOLVE = %s.etsolve\n\n' % model_name)
        if put_results is not None:
            if put_results_format == 'gdx':
                output_file.write("\nexecute_unload '%s_s.gdx'" % put_results)
                for stat in stat_vars:
                    output_file.write(', %s' % stat)
                output_file.write(';\n')
            else:
                results = put_results + '.dat'
                output_file.write("\nfile results /'%s'/;" % results)
                output_file.write('\nresults.nd=15;')
                output_file.write('\nresults.nw=21;')
                output_file.write('\nput results;')
                output_file.write("\nput 'SYMBOL  :  LEVEL  :  MARGINAL' /;")
                for var in var_list:
                    output_file.write("\nput %s ' ' %s.l ' ' %s.m /;" % (var, var, var))
                for con in constraint_names:
                    output_file.write("\nput %s ' ' %s.l ' ' %s.m /;" % (con, con, con))
                output_file.write("\nput GAMS_OBJECTIVE ' ' GAMS_OBJECTIVE.l ' ' GAMS_OBJECTIVE.m;\n")
                statresults = put_results + 'stat.dat'
                output_file.write("\nfile statresults /'%s'/;" % statresults)
                output_file.write('\nstatresults.nd=15;')
                output_file.write('\nstatresults.nw=21;')
                output_file.write('\nput statresults;')
                output_file.write("\nput 'SYMBOL   :   VALUE' /;")
                for stat in stat_vars:
                    output_file.write("\nput '%s' ' ' %s /;\n" % (stat, stat))