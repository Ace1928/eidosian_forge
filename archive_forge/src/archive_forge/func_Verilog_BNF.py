import time
import pprint
import sys
from pyparsing import Literal, Keyword, Word, OneOrMore, ZeroOrMore, \
import pyparsing
def Verilog_BNF():
    global verilogbnf
    if verilogbnf is None:
        compilerDirective = Combine('`' + oneOf('define undef ifdef else endif default_nettype include resetall timescale unconnected_drive nounconnected_drive celldefine endcelldefine') + restOfLine).setName('compilerDirective')
        SEMI, COLON, LPAR, RPAR, LBRACE, RBRACE, LBRACK, RBRACK, DOT, COMMA, EQ = map(Literal, ';:(){}[].,=')
        identLead = alphas + '$_'
        identBody = alphanums + '$_'
        identifier1 = Regex('\\.?[' + identLead + '][' + identBody + ']*(\\.[' + identLead + '][' + identBody + ']*)*').setName('baseIdent')
        identifier2 = Regex('\\\\\\S+').setParseAction(lambda t: t[0][1:]).setName('escapedIdent')
        identifier = identifier1 | identifier2
        assert identifier2 == '\\abc'
        hexnums = nums + 'abcdefABCDEF' + '_?'
        base = Regex("'[bBoOdDhH]").setName('base')
        basedNumber = Combine(Optional(Word(nums + '_')) + base + Word(hexnums + 'xXzZ'), joinString=' ', adjacent=False).setName('basedNumber')
        number = (basedNumber | Regex('[+-]?[0-9_]+(\\.[0-9_]*)?([Ee][+-]?[0-9_]+)?')).setName('numeric')
        expr = Forward().setName('expr')
        concat = Group(LBRACE + delimitedList(expr) + RBRACE)
        multiConcat = Group('{' + expr + concat + '}').setName('multiConcat')
        funcCall = Group(identifier + LPAR + Optional(delimitedList(expr)) + RPAR).setName('funcCall')
        subscrRef = Group(LBRACK + delimitedList(expr, COLON) + RBRACK)
        subscrIdentifier = Group(identifier + Optional(subscrRef))
        scalarConst = Regex("0|1('[Bb][01xX])?")
        mintypmaxExpr = Group(expr + COLON + expr + COLON + expr).setName('mintypmax')
        primary = number | LPAR + mintypmaxExpr + RPAR | (LPAR + Group(expr) + RPAR).setName('nestedExpr') | multiConcat | concat | dblQuotedString | funcCall | subscrIdentifier
        unop = oneOf('+  -  !  ~  &  ~&  |  ^|  ^  ~^').setName('unop')
        binop = oneOf('+  -  *  /  %  ==  !=  ===  !==  &&  ||  <  <=  >  >=  &  |  ^  ^~  >>  << ** <<< >>>').setName('binop')
        expr << (unop + expr | primary + '?' + expr + COLON + expr | primary + Optional(binop + expr))
        lvalue = subscrIdentifier | concat
        if_ = Keyword('if')
        else_ = Keyword('else')
        edge = Keyword('edge')
        posedge = Keyword('posedge')
        negedge = Keyword('negedge')
        specify = Keyword('specify')
        endspecify = Keyword('endspecify')
        fork = Keyword('fork')
        join = Keyword('join')
        begin = Keyword('begin')
        end = Keyword('end')
        default = Keyword('default')
        forever = Keyword('forever')
        repeat = Keyword('repeat')
        while_ = Keyword('while')
        for_ = Keyword('for')
        case = oneOf('case casez casex')
        endcase = Keyword('endcase')
        wait = Keyword('wait')
        disable = Keyword('disable')
        deassign = Keyword('deassign')
        force = Keyword('force')
        release = Keyword('release')
        assign = Keyword('assign')
        eventExpr = Forward()
        eventTerm = posedge + expr | negedge + expr | expr | LPAR + eventExpr + RPAR
        eventExpr << Group(delimitedList(eventTerm, Keyword('or')))
        eventControl = Group('@' + (LPAR + eventExpr + RPAR | identifier | '*')).setName('eventCtrl')
        delayArg = (number | Word(alphanums + '$_') | LPAR + Group(delimitedList(mintypmaxExpr | expr)) + RPAR).setName('delayArg')
        delay = Group('#' + delayArg).setName('delay')
        delayOrEventControl = delay | eventControl
        assgnmt = Group(lvalue + EQ + Optional(delayOrEventControl) + expr).setName('assgnmt')
        nbAssgnmt = Group(lvalue + '<=' + Optional(delay) + expr | lvalue + '<=' + Optional(eventControl) + expr).setName('nbassgnmt')
        range = LBRACK + expr + COLON + expr + RBRACK
        paramAssgnmt = Group(identifier + EQ + expr).setName('paramAssgnmt')
        parameterDecl = Group('parameter' + Optional(range) + delimitedList(paramAssgnmt) + SEMI).setName('paramDecl')
        inputDecl = Group('input' + Optional(range) + delimitedList(identifier) + SEMI)
        outputDecl = Group('output' + Optional(range) + delimitedList(identifier) + SEMI)
        inoutDecl = Group('inout' + Optional(range) + delimitedList(identifier) + SEMI)
        regIdentifier = Group(identifier + Optional(LBRACK + expr + COLON + expr + RBRACK))
        regDecl = Group('reg' + Optional('signed') + Optional(range) + delimitedList(regIdentifier) + SEMI).setName('regDecl')
        timeDecl = Group('time' + delimitedList(regIdentifier) + SEMI)
        integerDecl = Group('integer' + delimitedList(regIdentifier) + SEMI)
        strength0 = oneOf('supply0  strong0  pull0  weak0  highz0')
        strength1 = oneOf('supply1  strong1  pull1  weak1  highz1')
        driveStrength = Group(LPAR + (strength0 + COMMA + strength1 | strength1 + COMMA + strength0) + RPAR).setName('driveStrength')
        nettype = oneOf('wire  tri  tri1  supply0  wand  triand  tri0  supply1  wor  trior  trireg')
        expandRange = Optional(oneOf('scalared vectored')) + range
        realDecl = Group('real' + delimitedList(identifier) + SEMI)
        eventDecl = Group('event' + delimitedList(identifier) + SEMI)
        blockDecl = parameterDecl | regDecl | integerDecl | realDecl | timeDecl | eventDecl
        stmt = Forward().setName('stmt')
        stmtOrNull = stmt | SEMI
        caseItem = delimitedList(expr) + COLON + stmtOrNull | default + Optional(':') + stmtOrNull
        stmt << Group((begin + Group(ZeroOrMore(stmt)) + end).setName('begin-end') | (if_ + Group(LPAR + expr + RPAR) + stmtOrNull + Optional(else_ + stmtOrNull)).setName('if') | delayOrEventControl + stmtOrNull | case + LPAR + expr + RPAR + OneOrMore(caseItem) + endcase | forever + stmt | repeat + LPAR + expr + RPAR + stmt | while_ + LPAR + expr + RPAR + stmt | for_ + LPAR + assgnmt + SEMI + Group(expr) + SEMI + assgnmt + RPAR + stmt | fork + ZeroOrMore(stmt) + join | fork + COLON + identifier + ZeroOrMore(blockDecl) + ZeroOrMore(stmt) + end | wait + LPAR + expr + RPAR + stmtOrNull | '->' + identifier + SEMI | disable + identifier + SEMI | assign + assgnmt + SEMI | deassign + lvalue + SEMI | force + assgnmt + SEMI | release + lvalue + SEMI | (begin + COLON + identifier + ZeroOrMore(blockDecl) + ZeroOrMore(stmt) + end).setName('begin:label-end') | assgnmt + SEMI | nbAssgnmt + SEMI | Combine(Optional('$') + identifier) + Optional(LPAR + delimitedList(expr | empty) + RPAR) + SEMI).setName('stmtBody')
        '\n        x::=<blocking_assignment> ;\n        x||= <non_blocking_assignment> ;\n        x||= if ( <expression> ) <statement_or_null>\n        x||= if ( <expression> ) <statement_or_null> else <statement_or_null>\n        x||= case ( <expression> ) <case_item>+ endcase\n        x||= casez ( <expression> ) <case_item>+ endcase\n        x||= casex ( <expression> ) <case_item>+ endcase\n        x||= forever <statement>\n        x||= repeat ( <expression> ) <statement>\n        x||= while ( <expression> ) <statement>\n        x||= for ( <assignment> ; <expression> ; <assignment> ) <statement>\n        x||= <delay_or_event_control> <statement_or_null>\n        x||= wait ( <expression> ) <statement_or_null>\n        x||= -> <name_of_event> ;\n        x||= <seq_block>\n        x||= <par_block>\n        x||= <task_enable>\n        x||= <system_task_enable>\n        x||= disable <name_of_task> ;\n        x||= disable <name_of_block> ;\n        x||= assign <assignment> ;\n        x||= deassign <lvalue> ;\n        x||= force <assignment> ;\n        x||= release <lvalue> ;\n        '
        alwaysStmt = Group('always' + Optional(eventControl) + stmt).setName('alwaysStmt')
        initialStmt = Group('initial' + stmt).setName('initialStmt')
        chargeStrength = Group(LPAR + oneOf('small medium large') + RPAR).setName('chargeStrength')
        continuousAssign = Group(assign + Optional(driveStrength) + Optional(delay) + delimitedList(assgnmt) + SEMI).setName('continuousAssign')
        tfDecl = parameterDecl | inputDecl | outputDecl | inoutDecl | regDecl | timeDecl | integerDecl | realDecl
        functionDecl = Group('function' + Optional(range | 'integer' | 'real') + identifier + SEMI + Group(OneOrMore(tfDecl)) + Group(ZeroOrMore(stmt)) + 'endfunction')
        inputOutput = oneOf('input output')
        netDecl1Arg = nettype + Optional(expandRange) + Optional(delay) + Group(delimitedList(~inputOutput + identifier))
        netDecl2Arg = 'trireg' + Optional(chargeStrength) + Optional(expandRange) + Optional(delay) + Group(delimitedList(~inputOutput + identifier))
        netDecl3Arg = nettype + Optional(driveStrength) + Optional(expandRange) + Optional(delay) + Group(delimitedList(assgnmt))
        netDecl1 = Group(netDecl1Arg + SEMI).setName('netDecl1')
        netDecl2 = Group(netDecl2Arg + SEMI).setName('netDecl2')
        netDecl3 = Group(netDecl3Arg + SEMI).setName('netDecl3')
        gateType = oneOf('and  nand  or  nor xor  xnor buf  bufif0 bufif1 not  notif0 notif1  pulldown pullup nmos  rnmos pmos rpmos cmos rcmos   tran rtran  tranif0  rtranif0  tranif1 rtranif1')
        gateInstance = Optional(Group(identifier + Optional(range))) + LPAR + Group(delimitedList(expr)) + RPAR
        gateDecl = Group(gateType + Optional(driveStrength) + Optional(delay) + delimitedList(gateInstance) + SEMI)
        udpInstance = Group(Group(identifier + Optional(range | subscrRef)) + LPAR + Group(delimitedList(expr)) + RPAR)
        udpInstantiation = Group(identifier - Optional(driveStrength) + Optional(delay) + delimitedList(udpInstance) + SEMI).setName('udpInstantiation')
        parameterValueAssignment = Group(Literal('#') + LPAR + Group(delimitedList(expr)) + RPAR)
        namedPortConnection = Group(DOT + identifier + LPAR + expr + RPAR).setName('namedPortConnection')
        assert '.\\abc (abc )' == namedPortConnection
        modulePortConnection = expr | empty
        inst_args = Group(LPAR + (delimitedList(namedPortConnection) | delimitedList(modulePortConnection)) + RPAR).setName('inst_args')
        moduleInstance = Group(Group(identifier + Optional(range)) + inst_args).setName('moduleInstance')
        moduleInstantiation = Group(identifier + Optional(parameterValueAssignment) + delimitedList(moduleInstance).setName('moduleInstanceList') + SEMI).setName('moduleInstantiation')
        parameterOverride = Group('defparam' + delimitedList(paramAssgnmt) + SEMI)
        task = Group('task' + identifier + SEMI + ZeroOrMore(tfDecl) + stmtOrNull + 'endtask')
        specparamDecl = Group('specparam' + delimitedList(paramAssgnmt) + SEMI)
        pathDescr1 = Group(LPAR + subscrIdentifier + '=>' + subscrIdentifier + RPAR)
        pathDescr2 = Group(LPAR + Group(delimitedList(subscrIdentifier)) + '*>' + Group(delimitedList(subscrIdentifier)) + RPAR)
        pathDescr3 = Group(LPAR + Group(delimitedList(subscrIdentifier)) + '=>' + Group(delimitedList(subscrIdentifier)) + RPAR)
        pathDelayValue = Group(LPAR + Group(delimitedList(mintypmaxExpr | expr)) + RPAR | mintypmaxExpr | expr)
        pathDecl = Group((pathDescr1 | pathDescr2 | pathDescr3) + EQ + pathDelayValue + SEMI).setName('pathDecl')
        portConditionExpr = Forward()
        portConditionTerm = Optional(unop) + subscrIdentifier
        portConditionExpr << portConditionTerm + Optional(binop + portConditionExpr)
        polarityOp = oneOf('+ -')
        levelSensitivePathDecl1 = Group(if_ + Group(LPAR + portConditionExpr + RPAR) + subscrIdentifier + Optional(polarityOp) + '=>' + subscrIdentifier + EQ + pathDelayValue + SEMI)
        levelSensitivePathDecl2 = Group(if_ + Group(LPAR + portConditionExpr + RPAR) + LPAR + Group(delimitedList(subscrIdentifier)) + Optional(polarityOp) + '*>' + Group(delimitedList(subscrIdentifier)) + RPAR + EQ + pathDelayValue + SEMI)
        levelSensitivePathDecl = levelSensitivePathDecl1 | levelSensitivePathDecl2
        edgeIdentifier = posedge | negedge
        edgeSensitivePathDecl1 = Group(Optional(if_ + Group(LPAR + expr + RPAR)) + LPAR + Optional(edgeIdentifier) + subscrIdentifier + '=>' + LPAR + subscrIdentifier + Optional(polarityOp) + COLON + expr + RPAR + RPAR + EQ + pathDelayValue + SEMI)
        edgeSensitivePathDecl2 = Group(Optional(if_ + Group(LPAR + expr + RPAR)) + LPAR + Optional(edgeIdentifier) + subscrIdentifier + '*>' + LPAR + delimitedList(subscrIdentifier) + Optional(polarityOp) + COLON + expr + RPAR + RPAR + EQ + pathDelayValue + SEMI)
        edgeSensitivePathDecl = edgeSensitivePathDecl1 | edgeSensitivePathDecl2
        edgeDescr = oneOf('01 10 0x x1 1x x0').setName('edgeDescr')
        timCheckEventControl = Group(posedge | negedge | edge + LBRACK + delimitedList(edgeDescr) + RBRACK)
        timCheckCond = Forward()
        timCondBinop = oneOf('== === != !==')
        timCheckCondTerm = expr + timCondBinop + scalarConst | Optional('~') + expr
        timCheckCond << (LPAR + timCheckCond + RPAR | timCheckCondTerm)
        timCheckEvent = Group(Optional(timCheckEventControl) + subscrIdentifier + Optional('&&&' + timCheckCond))
        timCheckLimit = expr
        controlledTimingCheckEvent = Group(timCheckEventControl + subscrIdentifier + Optional('&&&' + timCheckCond))
        notifyRegister = identifier
        systemTimingCheck1 = Group('$setup' + LPAR + timCheckEvent + COMMA + timCheckEvent + COMMA + timCheckLimit + Optional(COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck2 = Group('$hold' + LPAR + timCheckEvent + COMMA + timCheckEvent + COMMA + timCheckLimit + Optional(COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck3 = Group('$period' + LPAR + controlledTimingCheckEvent + COMMA + timCheckLimit + Optional(COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck4 = Group('$width' + LPAR + controlledTimingCheckEvent + COMMA + timCheckLimit + Optional(COMMA + expr + COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck5 = Group('$skew' + LPAR + timCheckEvent + COMMA + timCheckEvent + COMMA + timCheckLimit + Optional(COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck6 = Group('$recovery' + LPAR + controlledTimingCheckEvent + COMMA + timCheckEvent + COMMA + timCheckLimit + Optional(COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck7 = Group('$setuphold' + LPAR + timCheckEvent + COMMA + timCheckEvent + COMMA + timCheckLimit + COMMA + timCheckLimit + Optional(COMMA + notifyRegister) + RPAR + SEMI)
        systemTimingCheck = (FollowedBy('$') + (systemTimingCheck1 | systemTimingCheck2 | systemTimingCheck3 | systemTimingCheck4 | systemTimingCheck5 | systemTimingCheck6 | systemTimingCheck7)).setName('systemTimingCheck')
        sdpd = if_ + Group(LPAR + expr + RPAR) + (pathDescr1 | pathDescr2) + EQ + pathDelayValue + SEMI
        specifyItem = ~Keyword('endspecify') + (specparamDecl | pathDecl | levelSensitivePathDecl | edgeSensitivePathDecl | systemTimingCheck | sdpd)
        '\n        x::= <specparam_declaration>\n        x||= <path_declaration>\n        x||= <level_sensitive_path_declaration>\n        x||= <edge_sensitive_path_declaration>\n        x||= <system_timing_check>\n        x||= <sdpd>\n        '
        specifyBlock = Group('specify' + ZeroOrMore(specifyItem) + 'endspecify').setName('specifyBlock')
        moduleItem = ~Keyword('endmodule') + (parameterDecl | inputDecl | outputDecl | inoutDecl | regDecl | netDecl3 | netDecl1 | netDecl2 | timeDecl | integerDecl | realDecl | eventDecl | gateDecl | parameterOverride | continuousAssign | specifyBlock | initialStmt | alwaysStmt | task | functionDecl | moduleInstantiation | udpInstantiation)
        '  All possible moduleItems, from Verilog grammar spec\n        x::= <parameter_declaration>\n        x||= <input_declaration>\n        x||= <output_declaration>\n        x||= <inout_declaration>\n        ?||= <net_declaration>  (spec does not seem consistent for this item)\n        x||= <reg_declaration>\n        x||= <time_declaration>\n        x||= <integer_declaration>\n        x||= <real_declaration>\n        x||= <event_declaration>\n        x||= <gate_declaration>\n        x||= <UDP_instantiation>\n        x||= <module_instantiation>\n        x||= <parameter_override>\n        x||= <continuous_assign>\n        x||= <specify_block>\n        x||= <initial_statement>\n        x||= <always_statement>\n        x||= <task>\n        x||= <function>\n        '
        portRef = subscrIdentifier
        portExpr = portRef | Group(LBRACE + delimitedList(portRef) + RBRACE)
        port = portExpr | Group(DOT + identifier + LPAR + portExpr + RPAR)
        moduleHdr = Group(oneOf('module macromodule') + identifier + Optional(LPAR + Group(Optional(delimitedList(Group(oneOf('input output') + (netDecl1Arg | netDecl2Arg | netDecl3Arg)) | port))) + RPAR) + SEMI).setName('moduleHdr')
        module = Group(moduleHdr + Group(ZeroOrMore(moduleItem)) + 'endmodule').setName('module')
        udpDecl = outputDecl | inputDecl | regDecl
        udpInitVal = (Regex("1'[bB][01xX]") | Regex('[01xX]')).setName('udpInitVal')
        udpInitialStmt = Group('initial' + identifier + EQ + udpInitVal + SEMI).setName('udpInitialStmt')
        levelSymbol = oneOf('0   1   x   X   ?   b   B')
        levelInputList = Group(OneOrMore(levelSymbol).setName('levelInpList'))
        outputSymbol = oneOf('0   1   x   X')
        combEntry = Group(levelInputList + COLON + outputSymbol + SEMI)
        edgeSymbol = oneOf('r   R   f   F   p   P   n   N   *')
        edge = Group(LPAR + levelSymbol + levelSymbol + RPAR) | Group(edgeSymbol)
        edgeInputList = Group(ZeroOrMore(levelSymbol) + edge + ZeroOrMore(levelSymbol))
        inputList = levelInputList | edgeInputList
        seqEntry = Group(inputList + COLON + levelSymbol + COLON + (outputSymbol | '-') + SEMI).setName('seqEntry')
        udpTableDefn = Group('table' + OneOrMore(combEntry | seqEntry) + 'endtable').setName('table')
        '\n        <UDP>\n        ::= primitive <name_of_UDP> ( <name_of_variable> <,<name_of_variable>>* ) ;\n                <UDP_declaration>+\n                <UDP_initial_statement>?\n                <table_definition>\n                endprimitive\n        '
        udp = Group('primitive' + identifier + LPAR + Group(delimitedList(identifier)) + RPAR + SEMI + OneOrMore(udpDecl) + Optional(udpInitialStmt) + udpTableDefn + 'endprimitive')
        verilogbnf = OneOrMore(module | udp) + StringEnd()
        verilogbnf.ignore(cppStyleComment)
        verilogbnf.ignore(compilerDirective)
    return verilogbnf