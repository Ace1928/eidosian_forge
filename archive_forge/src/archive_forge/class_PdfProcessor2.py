from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class PdfProcessor2(PdfProcessor):
    """ Wrapper class for struct pdf_processor with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == PdfProcessor2:
            _self = None
        else:
            _self = self
        _mupdf.PdfProcessor2_swiginit(self, _mupdf.new_PdfProcessor2(_self))

    def use_virtual_close_processor(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.PdfProcessor2_use_virtual_close_processor(self, use)

    def use_virtual_drop_processor(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_drop_processor(self, use)

    def use_virtual_push_resources(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_push_resources(self, use)

    def use_virtual_pop_resources(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_pop_resources(self, use)

    def use_virtual_op_w(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_w(self, use)

    def use_virtual_op_j(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_j(self, use)

    def use_virtual_op_J(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_J(self, use)

    def use_virtual_op_M(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_M(self, use)

    def use_virtual_op_d(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_d(self, use)

    def use_virtual_op_ri(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_ri(self, use)

    def use_virtual_op_i(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_i(self, use)

    def use_virtual_op_gs_begin(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_begin(self, use)

    def use_virtual_op_gs_BM(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_BM(self, use)

    def use_virtual_op_gs_ca(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_ca(self, use)

    def use_virtual_op_gs_CA(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_CA(self, use)

    def use_virtual_op_gs_SMask(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_SMask(self, use)

    def use_virtual_op_gs_end(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_end(self, use)

    def use_virtual_op_q(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_q(self, use)

    def use_virtual_op_Q(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Q(self, use)

    def use_virtual_op_cm(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_cm(self, use)

    def use_virtual_op_m(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_m(self, use)

    def use_virtual_op_l(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_l(self, use)

    def use_virtual_op_c(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_c(self, use)

    def use_virtual_op_v(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_v(self, use)

    def use_virtual_op_y(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_y(self, use)

    def use_virtual_op_h(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_h(self, use)

    def use_virtual_op_re(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_re(self, use)

    def use_virtual_op_S(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_S(self, use)

    def use_virtual_op_s(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_s(self, use)

    def use_virtual_op_F(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_F(self, use)

    def use_virtual_op_f(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_f(self, use)

    def use_virtual_op_fstar(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_fstar(self, use)

    def use_virtual_op_B(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_B(self, use)

    def use_virtual_op_Bstar(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Bstar(self, use)

    def use_virtual_op_b(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_b(self, use)

    def use_virtual_op_bstar(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_bstar(self, use)

    def use_virtual_op_n(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_n(self, use)

    def use_virtual_op_W(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_W(self, use)

    def use_virtual_op_Wstar(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Wstar(self, use)

    def use_virtual_op_BT(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_BT(self, use)

    def use_virtual_op_ET(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_ET(self, use)

    def use_virtual_op_Tc(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tc(self, use)

    def use_virtual_op_Tw(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tw(self, use)

    def use_virtual_op_Tz(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tz(self, use)

    def use_virtual_op_TL(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_TL(self, use)

    def use_virtual_op_Tf(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tf(self, use)

    def use_virtual_op_Tr(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tr(self, use)

    def use_virtual_op_Ts(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Ts(self, use)

    def use_virtual_op_Td(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Td(self, use)

    def use_virtual_op_TD(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_TD(self, use)

    def use_virtual_op_Tm(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tm(self, use)

    def use_virtual_op_Tstar(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tstar(self, use)

    def use_virtual_op_TJ(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_TJ(self, use)

    def use_virtual_op_Tj(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Tj(self, use)

    def use_virtual_op_squote(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_squote(self, use)

    def use_virtual_op_dquote(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_dquote(self, use)

    def use_virtual_op_d0(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_d0(self, use)

    def use_virtual_op_d1(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_d1(self, use)

    def use_virtual_op_CS(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_CS(self, use)

    def use_virtual_op_cs(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_cs(self, use)

    def use_virtual_op_SC_pattern(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_SC_pattern(self, use)

    def use_virtual_op_sc_pattern(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_sc_pattern(self, use)

    def use_virtual_op_SC_shade(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_SC_shade(self, use)

    def use_virtual_op_sc_shade(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_sc_shade(self, use)

    def use_virtual_op_SC_color(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_SC_color(self, use)

    def use_virtual_op_sc_color(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_sc_color(self, use)

    def use_virtual_op_G(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_G(self, use)

    def use_virtual_op_g(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_g(self, use)

    def use_virtual_op_RG(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_RG(self, use)

    def use_virtual_op_rg(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_rg(self, use)

    def use_virtual_op_K(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_K(self, use)

    def use_virtual_op_k(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_k(self, use)

    def use_virtual_op_BI(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_BI(self, use)

    def use_virtual_op_sh(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_sh(self, use)

    def use_virtual_op_Do_image(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Do_image(self, use)

    def use_virtual_op_Do_form(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_Do_form(self, use)

    def use_virtual_op_MP(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_MP(self, use)

    def use_virtual_op_DP(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_DP(self, use)

    def use_virtual_op_BMC(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_BMC(self, use)

    def use_virtual_op_BDC(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_BDC(self, use)

    def use_virtual_op_EMC(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_EMC(self, use)

    def use_virtual_op_BX(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_BX(self, use)

    def use_virtual_op_EX(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_EX(self, use)

    def use_virtual_op_gs_OP(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_OP(self, use)

    def use_virtual_op_gs_op(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_op(self, use)

    def use_virtual_op_gs_OPM(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_OPM(self, use)

    def use_virtual_op_gs_UseBlackPtComp(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_gs_UseBlackPtComp(self, use)

    def use_virtual_op_END(self, use=True):
        return _mupdf.PdfProcessor2_use_virtual_op_END(self, use)

    def close_processor(self, arg_0):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.PdfProcessor2_close_processor(self, arg_0)

    def drop_processor(self, arg_0):
        return _mupdf.PdfProcessor2_drop_processor(self, arg_0)

    def push_resources(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_push_resources(self, arg_0, arg_2)

    def pop_resources(self, arg_0):
        return _mupdf.PdfProcessor2_pop_resources(self, arg_0)

    def op_w(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_w(self, arg_0, arg_2)

    def op_j(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_j(self, arg_0, arg_2)

    def op_J(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_J(self, arg_0, arg_2)

    def op_M(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_M(self, arg_0, arg_2)

    def op_d(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_d(self, arg_0, arg_2, arg_3)

    def op_ri(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_ri(self, arg_0, arg_2)

    def op_i(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_i(self, arg_0, arg_2)

    def op_gs_begin(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_gs_begin(self, arg_0, arg_2, arg_3)

    def op_gs_BM(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_BM(self, arg_0, arg_2)

    def op_gs_ca(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_ca(self, arg_0, arg_2)

    def op_gs_CA(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_CA(self, arg_0, arg_2)

    def op_gs_SMask(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_gs_SMask(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_gs_end(self, arg_0):
        return _mupdf.PdfProcessor2_op_gs_end(self, arg_0)

    def op_q(self, arg_0):
        return _mupdf.PdfProcessor2_op_q(self, arg_0)

    def op_Q(self, arg_0):
        return _mupdf.PdfProcessor2_op_Q(self, arg_0)

    def op_cm(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.PdfProcessor2_op_cm(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def op_m(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_m(self, arg_0, arg_2, arg_3)

    def op_l(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_l(self, arg_0, arg_2, arg_3)

    def op_c(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.PdfProcessor2_op_c(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def op_v(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_v(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_y(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_y(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_h(self, arg_0):
        return _mupdf.PdfProcessor2_op_h(self, arg_0)

    def op_re(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_re(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_S(self, arg_0):
        return _mupdf.PdfProcessor2_op_S(self, arg_0)

    def op_s(self, arg_0):
        return _mupdf.PdfProcessor2_op_s(self, arg_0)

    def op_F(self, arg_0):
        return _mupdf.PdfProcessor2_op_F(self, arg_0)

    def op_f(self, arg_0):
        return _mupdf.PdfProcessor2_op_f(self, arg_0)

    def op_fstar(self, arg_0):
        return _mupdf.PdfProcessor2_op_fstar(self, arg_0)

    def op_B(self, arg_0):
        return _mupdf.PdfProcessor2_op_B(self, arg_0)

    def op_Bstar(self, arg_0):
        return _mupdf.PdfProcessor2_op_Bstar(self, arg_0)

    def op_b(self, arg_0):
        return _mupdf.PdfProcessor2_op_b(self, arg_0)

    def op_bstar(self, arg_0):
        return _mupdf.PdfProcessor2_op_bstar(self, arg_0)

    def op_n(self, arg_0):
        return _mupdf.PdfProcessor2_op_n(self, arg_0)

    def op_W(self, arg_0):
        return _mupdf.PdfProcessor2_op_W(self, arg_0)

    def op_Wstar(self, arg_0):
        return _mupdf.PdfProcessor2_op_Wstar(self, arg_0)

    def op_BT(self, arg_0):
        return _mupdf.PdfProcessor2_op_BT(self, arg_0)

    def op_ET(self, arg_0):
        return _mupdf.PdfProcessor2_op_ET(self, arg_0)

    def op_Tc(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_Tc(self, arg_0, arg_2)

    def op_Tw(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_Tw(self, arg_0, arg_2)

    def op_Tz(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_Tz(self, arg_0, arg_2)

    def op_TL(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_TL(self, arg_0, arg_2)

    def op_Tf(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.PdfProcessor2_op_Tf(self, arg_0, arg_2, arg_3, arg_4)

    def op_Tr(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_Tr(self, arg_0, arg_2)

    def op_Ts(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_Ts(self, arg_0, arg_2)

    def op_Td(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_Td(self, arg_0, arg_2, arg_3)

    def op_TD(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_TD(self, arg_0, arg_2, arg_3)

    def op_Tm(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.PdfProcessor2_op_Tm(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def op_Tstar(self, arg_0):
        return _mupdf.PdfProcessor2_op_Tstar(self, arg_0)

    def op_TJ(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_TJ(self, arg_0, arg_2)

    def op_Tj(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_Tj(self, arg_0, arg_2, arg_3)

    def op_squote(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_squote(self, arg_0, arg_2, arg_3)

    def op_dquote(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_dquote(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_d0(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_d0(self, arg_0, arg_2, arg_3)

    def op_d1(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.PdfProcessor2_op_d1(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def op_CS(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_CS(self, arg_0, arg_2, arg_3)

    def op_cs(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_cs(self, arg_0, arg_2, arg_3)

    def op_SC_pattern(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_SC_pattern(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_sc_pattern(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_sc_pattern(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_SC_shade(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_SC_shade(self, arg_0, arg_2, arg_3)

    def op_sc_shade(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_sc_shade(self, arg_0, arg_2, arg_3)

    def op_SC_color(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_SC_color(self, arg_0, arg_2, arg_3)

    def op_sc_color(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_sc_color(self, arg_0, arg_2, arg_3)

    def op_G(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_G(self, arg_0, arg_2)

    def op_g(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_g(self, arg_0, arg_2)

    def op_RG(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.PdfProcessor2_op_RG(self, arg_0, arg_2, arg_3, arg_4)

    def op_rg(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.PdfProcessor2_op_rg(self, arg_0, arg_2, arg_3, arg_4)

    def op_K(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_K(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_k(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.PdfProcessor2_op_k(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def op_BI(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_BI(self, arg_0, arg_2, arg_3)

    def op_sh(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_sh(self, arg_0, arg_2, arg_3)

    def op_Do_image(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_Do_image(self, arg_0, arg_2, arg_3)

    def op_Do_form(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfProcessor2_op_Do_form(self, arg_0, arg_2, arg_3)

    def op_MP(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_MP(self, arg_0, arg_2)

    def op_DP(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.PdfProcessor2_op_DP(self, arg_0, arg_2, arg_3, arg_4)

    def op_BMC(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_BMC(self, arg_0, arg_2)

    def op_BDC(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.PdfProcessor2_op_BDC(self, arg_0, arg_2, arg_3, arg_4)

    def op_EMC(self, arg_0):
        return _mupdf.PdfProcessor2_op_EMC(self, arg_0)

    def op_BX(self, arg_0):
        return _mupdf.PdfProcessor2_op_BX(self, arg_0)

    def op_EX(self, arg_0):
        return _mupdf.PdfProcessor2_op_EX(self, arg_0)

    def op_gs_OP(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_OP(self, arg_0, arg_2)

    def op_gs_op(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_op(self, arg_0, arg_2)

    def op_gs_OPM(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_OPM(self, arg_0, arg_2)

    def op_gs_UseBlackPtComp(self, arg_0, arg_2):
        return _mupdf.PdfProcessor2_op_gs_UseBlackPtComp(self, arg_0, arg_2)

    def op_END(self, arg_0):
        return _mupdf.PdfProcessor2_op_END(self, arg_0)
    __swig_destroy__ = _mupdf.delete_PdfProcessor2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_PdfProcessor2(self)
        return weakref.proxy(self)