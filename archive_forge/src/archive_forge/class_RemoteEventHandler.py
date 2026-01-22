import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
class RemoteEventHandler(object):
    """
    This class handles communication between two processes. One instance is present on 
    each process and listens for communication from the other process. This enables
    (amongst other things) ObjectProxy instances to look up their attributes and call 
    their methods.
    
    This class is responsible for carrying out actions on behalf of the remote process.
    Each instance holds one end of a Connection which allows python
    objects to be passed between processes.
    
    For the most common operations, see _import(), close(), and transfer()
    
    To handle and respond to incoming requests, RemoteEventHandler requires that its
    processRequests method is called repeatedly (this is usually handled by the Process
    classes defined in multiprocess.processes).
    
    
    
    
    """
    handlers = {}

    def __init__(self, connection, name, pid, debug=False):
        self.debug = debug
        self.conn = connection
        self.name = name
        self.results = {}
        self.resultLock = threading.RLock()
        self.proxies = {}
        self.proxyLock = threading.RLock()
        self.proxyOptions = {'callSync': 'sync', 'timeout': 10, 'returnType': 'auto', 'autoProxy': False, 'deferGetattr': False, 'noProxyTypes': [type(None), str, bytes, int, float, tuple, list, dict, LocalObjectProxy, ObjectProxy]}
        self.optsLock = threading.RLock()
        self.nextRequestId = 0
        self.exited = False
        self.processLock = threading.RLock()
        self.sendLock = threading.RLock()
        if pid is None:
            connection.send(os.getpid())
            pid = connection.recv()
        RemoteEventHandler.handlers[pid] = self

    @classmethod
    def getHandler(cls, pid):
        try:
            return cls.handlers[pid]
        except:
            print(pid, cls.handlers)
            raise

    def debugMsg(self, msg, *args):
        if not self.debug:
            return
        cprint.cout(self.debug, '[%d] %s\n' % (os.getpid(), str(msg) % args), -1)

    def getProxyOption(self, opt):
        with self.optsLock:
            return self.proxyOptions[opt]

    def setProxyOptions(self, **kwds):
        """
        Set the default behavior options for object proxies.
        See ObjectProxy._setProxyOptions for more info.
        """
        with self.optsLock:
            self.proxyOptions.update(kwds)

    def processRequests(self):
        """Process all pending requests from the pipe, return
        after no more events are immediately available. (non-blocking)
        Returns the number of events processed.
        """
        with self.processLock:
            if self.exited:
                self.debugMsg('  processRequests: exited already; raise ClosedError.')
                raise ClosedError()
            numProcessed = 0
            while self.conn.poll():
                try:
                    self.handleRequest()
                    numProcessed += 1
                except ClosedError:
                    self.debugMsg('processRequests: got ClosedError from handleRequest; setting exited=True.')
                    self.exited = True
                    raise
                except:
                    print('Error in process %s' % self.name)
                    sys.excepthook(*sys.exc_info())
            if numProcessed > 0:
                self.debugMsg('processRequests: finished %d requests', numProcessed)
            return numProcessed

    def handleRequest(self):
        """Handle a single request from the remote process. 
        Blocks until a request is available."""
        result = None
        while True:
            try:
                cmd, reqId, nByteMsgs, optStr = self.conn.recv()
                break
            except EOFError:
                self.debugMsg('  handleRequest: got EOFError from recv; raise ClosedError.')
                raise ClosedError()
            except IOError as err:
                if err.errno == 4:
                    self.debugMsg('  handleRequest: got IOError 4 from recv; try again.')
                    continue
                else:
                    self.debugMsg('  handleRequest: got IOError %d from recv (%s); raise ClosedError.', err.errno, err.strerror)
                    raise ClosedError()
        self.debugMsg('  handleRequest: received %s %s', cmd, reqId)
        byteData = []
        if nByteMsgs > 0:
            self.debugMsg('    handleRequest: reading %d byte messages', nByteMsgs)
        for i in range(nByteMsgs):
            while True:
                try:
                    byteData.append(self.conn.recv_bytes())
                    break
                except EOFError:
                    self.debugMsg('    handleRequest: got EOF while reading byte messages; raise ClosedError.')
                    raise ClosedError()
                except IOError as err:
                    if err.errno == 4:
                        self.debugMsg('    handleRequest: got IOError 4 while reading byte messages; try again.')
                        continue
                    else:
                        self.debugMsg('    handleRequest: got IOError while reading byte messages; raise ClosedError.')
                        raise ClosedError()
        try:
            if cmd == 'result' or cmd == 'error':
                resultId = reqId
                reqId = None
            opts = pickle.loads(optStr)
            self.debugMsg('    handleRequest: id=%s opts=%s', reqId, opts)
            returnType = opts.get('returnType', 'auto')
            if cmd == 'result':
                with self.resultLock:
                    self.results[resultId] = ('result', opts['result'])
            elif cmd == 'error':
                with self.resultLock:
                    self.results[resultId] = ('error', (opts['exception'], opts['excString']))
            elif cmd == 'getObjAttr':
                result = getattr(opts['obj'], opts['attr'])
            elif cmd == 'callObj':
                obj = opts['obj']
                fnargs = opts['args']
                fnkwds = opts['kwds']
                if len(byteData) > 0:
                    for i, arg in enumerate(fnargs):
                        if isinstance(arg, tuple) and len(arg) > 0 and (arg[0] == '__byte_message__'):
                            ind = arg[1]
                            dtype, shape = arg[2]
                            fnargs[i] = np.frombuffer(byteData[ind], dtype=dtype).reshape(shape)
                    for k, arg in fnkwds.items():
                        if isinstance(arg, tuple) and len(arg) > 0 and (arg[0] == '__byte_message__'):
                            ind = arg[1]
                            dtype, shape = arg[2]
                            fnkwds[k] = np.frombuffer(byteData[ind], dtype=dtype).reshape(shape)
                if len(fnkwds) == 0:
                    try:
                        result = obj(*fnargs)
                    except:
                        print('Failed to call object %s: %d, %s' % (obj, len(fnargs), fnargs[1:]))
                        raise
                else:
                    result = obj(*fnargs, **fnkwds)
            elif cmd == 'getObjValue':
                result = opts['obj']
                returnType = 'value'
            elif cmd == 'transfer':
                result = opts['obj']
                returnType = 'proxy'
            elif cmd == 'transferArray':
                result = np.frombuffer(byteData[0], dtype=opts['dtype']).reshape(opts['shape'])
                returnType = 'proxy'
            elif cmd == 'import':
                name = opts['module']
                fromlist = opts.get('fromlist', [])
                mod = builtins.__import__(name, fromlist=fromlist)
                if len(fromlist) == 0:
                    parts = name.lstrip('.').split('.')
                    result = mod
                    for part in parts[1:]:
                        result = getattr(result, part)
                else:
                    result = map(mod.__getattr__, fromlist)
            elif cmd == 'del':
                LocalObjectProxy.releaseProxyId(opts['proxyId'])
            elif cmd == 'close':
                if reqId is not None:
                    result = True
                    returnType = 'value'
            exc = None
        except:
            exc = sys.exc_info()
        if reqId is not None:
            if exc is None:
                self.debugMsg('    handleRequest: sending return value for %d: %s', reqId, result)
                if returnType == 'auto':
                    with self.optsLock:
                        noProxyTypes = self.proxyOptions['noProxyTypes']
                    result = self.autoProxy(result, noProxyTypes)
                elif returnType == 'proxy':
                    result = LocalObjectProxy(result)
                try:
                    self.replyResult(reqId, result)
                except:
                    sys.excepthook(*sys.exc_info())
                    self.replyError(reqId, *sys.exc_info())
            else:
                self.debugMsg('    handleRequest: returning exception for %d', reqId)
                self.replyError(reqId, *exc)
        elif exc is not None:
            sys.excepthook(*exc)
        if cmd == 'close':
            if opts.get('noCleanup', False) is True:
                os._exit(0)
            else:
                raise ClosedError()

    def replyResult(self, reqId, result):
        self.send(request='result', reqId=reqId, callSync='off', opts=dict(result=result))

    def replyError(self, reqId, *exc):
        excStr = traceback.format_exception(*exc)
        try:
            self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=exc[1], excString=excStr))
        except:
            self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=None, excString=excStr))

    def send(self, request, opts=None, reqId=None, callSync='sync', timeout=10, returnType=None, byteData=None, **kwds):
        """Send a request or return packet to the remote process.
        Generally it is not necessary to call this method directly; it is for internal use.
        (The docstring has information that is nevertheless useful to the programmer
        as it describes the internal protocol used to communicate between processes)
        
        ==============  ====================================================================
        **Arguments:**
        request         String describing the type of request being sent (see below)
        reqId           Integer uniquely linking a result back to the request that generated
                        it. (most requests leave this blank)
        callSync        'sync':  return the actual result of the request
                        'async': return a Request object which can be used to look up the
                                result later
                        'off':   return no result
        timeout         Time in seconds to wait for a response when callSync=='sync'
        opts            Extra arguments sent to the remote process that determine the way
                        the request will be handled (see below)
        returnType      'proxy', 'value', or 'auto'
        byteData        If specified, this is a list of objects to be sent as byte messages
                        to the remote process.
                        This is used to send large arrays without the cost of pickling.
        ==============  ====================================================================
        
        Description of request strings and options allowed for each:
        
        =============  =============  ========================================================
        request        option         description
        -------------  -------------  --------------------------------------------------------
        getObjAttr                    Request the remote process return (proxy to) an
                                      attribute of an object.
                       obj            reference to object whose attribute should be 
                                      returned
                       attr           string name of attribute to return
                       returnValue    bool or 'auto' indicating whether to return a proxy or
                                      the actual value. 
                       
        callObj                       Request the remote process call a function or 
                                      method. If a request ID is given, then the call's
                                      return value will be sent back (or information
                                      about the error that occurred while running the
                                      function)
                       obj            the (reference to) object to call
                       args           tuple of arguments to pass to callable
                       kwds           dict of keyword arguments to pass to callable
                       returnValue    bool or 'auto' indicating whether to return a proxy or
                                      the actual value. 
                       
        getObjValue                   Request the remote process return the value of
                                      a proxied object (must be picklable)
                       obj            reference to object whose value should be returned
                       
        transfer                      Copy an object to the remote process and request
                                      it return a proxy for the new object.
                       obj            The object to transfer.
                       
        import                        Request the remote process import new symbols
                                      and return proxy(ies) to the imported objects
                       module         the string name of the module to import
                       fromlist       optional list of string names to import from module
                       
        del                           Inform the remote process that a proxy has been 
                                      released (thus the remote process may be able to 
                                      release the original object)
                       proxyId        id of proxy which is no longer referenced by 
                                      remote host
                                      
        close                         Instruct the remote process to stop its event loop
                                      and exit. Optionally, this request may return a 
                                      confirmation.
            
        result                        Inform the remote process that its request has 
                                      been processed                        
                       result         return value of a request
                       
        error                         Inform the remote process that its request failed
                       exception      the Exception that was raised (or None if the 
                                      exception could not be pickled)
                       excString      string-formatted version of the exception and 
                                      traceback
        =============  =====================================================================
        """
        if self.exited:
            self.debugMsg('  send: exited already; raise ClosedError.')
            raise ClosedError()
        with self.sendLock:
            if opts is None:
                opts = {}
            assert callSync in ['off', 'sync', 'async'], 'callSync must be one of "off", "sync", or "async" (got %r)' % callSync
            if reqId is None:
                if callSync != 'off':
                    reqId = self.nextRequestId
                    self.nextRequestId += 1
            else:
                assert request in ['result', 'error']
            if returnType is not None:
                opts['returnType'] = returnType
            try:
                optStr = pickle.dumps(opts)
            except:
                print('====  Error pickling this object:  ====')
                print(opts)
                print('=======================================')
                raise
            nByteMsgs = 0
            if byteData is not None:
                nByteMsgs = len(byteData)
            request = (request, reqId, nByteMsgs, optStr)
            self.debugMsg('send request: cmd=%s nByteMsgs=%d id=%s opts=%s', request[0], nByteMsgs, reqId, opts)
            self.conn.send(request)
            if byteData is not None:
                for obj in byteData:
                    self.conn.send_bytes(bytes(obj))
                self.debugMsg('  sent %d byte messages', len(byteData))
            self.debugMsg('  call sync: %s', callSync)
            if callSync == 'off':
                return
        req = Request(self, reqId, description=str(request), timeout=timeout)
        if callSync == 'async':
            return req
        if callSync == 'sync':
            return req.result()

    def close(self, callSync='off', noCleanup=False, **kwds):
        try:
            self.send(request='close', opts=dict(noCleanup=noCleanup), callSync=callSync, **kwds)
            self.exited = True
        except ClosedError:
            pass

    def getResult(self, reqId):
        with self.resultLock:
            haveResult = reqId in self.results
        if not haveResult:
            try:
                self.processRequests()
            except ClosedError:
                pass
        with self.resultLock:
            if reqId not in self.results:
                raise NoResultError()
            status, result = self.results.pop(reqId)
        if status == 'result':
            return result
        elif status == 'error':
            exc, excStr = result
            if exc is not None:
                normal = ['AttributeError']
                if not any((excStr[-1].startswith(x) for x in normal)):
                    warnings.warn('===== Remote process raised exception on request: =====', RemoteExceptionWarning)
                    warnings.warn(''.join(excStr), RemoteExceptionWarning)
                    warnings.warn('===== Local Traceback to request follows: =====', RemoteExceptionWarning)
                raise exc
            else:
                print(''.join(excStr))
                raise Exception('Error getting result. See above for exception from remote process.')
        else:
            raise Exception('Internal error.')

    def _import(self, mod, **kwds):
        """
        Request the remote process import a module (or symbols from a module)
        and return the proxied results. Uses built-in __import__() function, but 
        adds a bit more processing:
        
            _import('module')  =>  returns module
            _import('module.submodule')  =>  returns submodule 
                                             (note this differs from behavior of __import__)
            _import('module', fromlist=[name1, name2, ...])  =>  returns [module.name1, module.name2, ...]
                                             (this also differs from behavior of __import__)
            
        """
        return self.send(request='import', callSync='sync', opts=dict(module=mod), **kwds)

    def getObjAttr(self, obj, attr, **kwds):
        return self.send(request='getObjAttr', opts=dict(obj=obj, attr=attr), **kwds)

    def getObjValue(self, obj, **kwds):
        return self.send(request='getObjValue', opts=dict(obj=obj), **kwds)

    def callObj(self, obj, args, kwds, **opts):
        opts = opts.copy()
        args = list(args)
        with self.optsLock:
            noProxyTypes = opts.pop('noProxyTypes', None)
            if noProxyTypes is None:
                noProxyTypes = self.proxyOptions['noProxyTypes']
            autoProxy = opts.pop('autoProxy', self.proxyOptions['autoProxy'])
        if autoProxy is True:
            args = [self.autoProxy(v, noProxyTypes) for v in args]
            for k, v in kwds.items():
                opts[k] = self.autoProxy(v, noProxyTypes)
        byteMsgs = []
        for i, arg in enumerate(args):
            if arg.__class__ == np.ndarray:
                args[i] = ('__byte_message__', len(byteMsgs), (arg.dtype, arg.shape))
                byteMsgs.append(arg)
        for k, v in kwds.items():
            if v.__class__ == np.ndarray:
                kwds[k] = ('__byte_message__', len(byteMsgs), (v.dtype, v.shape))
                byteMsgs.append(v)
        return self.send(request='callObj', opts=dict(obj=obj, args=args, kwds=kwds), byteData=byteMsgs, **opts)

    def registerProxy(self, proxy):
        with self.proxyLock:
            ref = weakref.ref(proxy, self.deleteProxy)
            self.proxies[ref] = proxy._proxyId

    def deleteProxy(self, ref):
        if self.send is None:
            return
        with self.proxyLock:
            proxyId = self.proxies.pop(ref)
        try:
            self.send(request='del', opts=dict(proxyId=proxyId), callSync='off')
        except ClosedError:
            pass

    def transfer(self, obj, **kwds):
        """
        Transfer an object by value to the remote host (the object must be picklable) 
        and return a proxy for the new remote object.
        """
        if obj.__class__ is np.ndarray:
            opts = {'dtype': obj.dtype, 'shape': obj.shape}
            return self.send(request='transferArray', opts=opts, byteData=[obj], **kwds)
        else:
            return self.send(request='transfer', opts=dict(obj=obj), **kwds)

    def autoProxy(self, obj, noProxyTypes):
        for typ in noProxyTypes:
            if isinstance(obj, typ):
                return obj
        return LocalObjectProxy(obj)