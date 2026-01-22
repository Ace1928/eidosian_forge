import numpy as np

function jc = c_sjt(n,p)
% PURPOSE: find critical values for Johansen trace statistic
% ------------------------------------------------------------
% USAGE:  jc = c_sjt(n,p)
% where:    n = dimension of the VAR system
%               NOTE: routine does not work for n > 12
%           p = order of time polynomial in the null-hypothesis
%                 p = -1, no deterministic part
%                 p =  0, for constant term
%                 p =  1, for constant plus time-trend
%                 p >  1  returns no critical values
% ------------------------------------------------------------
% RETURNS: a (3x1) vector of percentiles for the trace
%          statistic for [90% 95% 99%]
% ------------------------------------------------------------
% NOTES: for n > 12, the function returns a (3x1) vector of zeros.
%        The values returned by the function were generated using
%        a method described in MacKinnon (1996), using his FORTRAN
%        program johdist.f
% ------------------------------------------------------------
% SEE ALSO: johansen()
% ------------------------------------------------------------
% % References: MacKinnon, Haug, Michelis (1996) 'Numerical distribution
% functions of likelihood ratio tests for cointegration',
% Queen's University Institute for Economic Research Discussion paper.
% -------------------------------------------------------
% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com
% these are the values from Johansen's 1995 book
% for comparison to the MacKinnon values
%jcp0 = [ 2.98   4.14   7.02
%        10.35  12.21  16.16
%        21.58  24.08  29.19
%        36.58  39.71  46.00
%        55.54  59.24  66.71
%        78.30  86.36  91.12
%       104.93 109.93 119.58
%       135.16 140.74 151.70
%       169.30 175.47 187.82
%       207.21 214.07 226.95
%       248.77 256.23 270.47
%       293.83 301.95 318.14];
%
