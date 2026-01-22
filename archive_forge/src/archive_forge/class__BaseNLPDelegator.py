from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
class _BaseNLPDelegator(NLP):

    def __init__(self, original_nlp):
        """
        This is a base class to make it easier to implement NLP
        classes that wrap other NLP instances. This class
        simply reproduces the NLP interface by passing the call
        onto the original_nlp passed in the constructor. This allows
        new wrapper classes to only implement the methods that change,
        allowing the others to pass through.

        Parameters
        ----------
        original_nlp : NLP-like
            The original NLP object that we want to wrap
        """
        super(NLP, self).__init__()
        self._original_nlp = original_nlp

    def n_primals(self):
        return self._original_nlp.n_primals()

    def primals_names(self):
        return self._original_nlp.primals_names()

    def n_constraints(self):
        return self._original_nlp.n_constraints()

    def constraint_names(self):
        return self._original_nlp.constraint_names()

    def nnz_jacobian(self):
        return self._original_nlp.nnz_jacobian()

    def nnz_hessian_lag(self):
        return self._original_nlp.nnz_hessian_lag()

    def primals_lb(self):
        return self._original_nlp.primals_lb()

    def primals_ub(self):
        return self._original_nlp.primals_ub()

    def constraints_lb(self):
        return self._original_nlp.constraints_lb()

    def constraints_ub(self):
        return self._original_nlp.constraints_ub()

    def init_primals(self):
        return self._original_nlp.init_primals()

    def init_duals(self):
        return self._original_nlp.init_duals()

    def create_new_vector(self, vector_type):
        return self._original_nlp.create_new_vector(vector_type)

    def set_primals(self, primals):
        self._original_nlp.set_primals(primals)

    def get_primals(self):
        return self._original_nlp.get_primals()

    def set_duals(self, duals):
        self._original_nlp.set_duals(duals)

    def get_duals(self):
        return self._original_nlp.get_duals()

    def set_obj_factor(self, obj_factor):
        self._original_nlp.set_obj_factor(obj_factor)

    def get_obj_factor(self):
        return self._original_nlp.get_obj_factor()

    def get_obj_scaling(self):
        return self._original_nlp.get_obj_scaling()

    def get_primals_scaling(self):
        return self._original_nlp.get_primals_scaling()

    def get_constraints_scaling(self):
        return self._original_nlp.get_constraints_scaling()

    def evaluate_objective(self):
        return self._original_nlp.evaluate_objective()

    def evaluate_grad_objective(self, out=None):
        return self._original_nlp.evaluate_grad_objective(out)

    def evaluate_constraints(self, out=None):
        return self._original_nlp.evaluate_constraints(out)

    def evaluate_jacobian(self, out=None):
        return self._original_nlp.evaluate_jacobian(out)

    def evaluate_hessian_lag(self, out=None):
        return self._original_nlp.evaluate_hessian_lag(out)

    def report_solver_status(self, status_code, status_message):
        self._original_nlp.report_solver_status(status_code, status_message)