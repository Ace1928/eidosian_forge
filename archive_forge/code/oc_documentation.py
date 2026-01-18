from pyparsing import *

https://www.ioccc.org/1996/august.hint

The following is a description of the OC grammar:

    OC grammar
    ==========
    Terminals are in quotes, () is used for bracketing.

    program:	decl*

    decl:		vardecl
            fundecl

    vardecl:	type NAME ;
            type NAME "[" INT "]" ;

    fundecl:	type NAME "(" args ")" "{" body "}"

    args:		/*empty*/
            ( arg "," )* arg

    arg:		type NAME

    body:		vardecl* stmt*

    stmt:		ifstmt
            whilestmt
            dowhilestmt
            "return" expr ";"
            expr ";"
            "{" stmt* "}"
            ";"

    ifstmt:		"if" "(" expr ")" stmt
            "if" "(" expr ")" stmt "else" stmt

    whilestmt:	"while" "(" expr ")" stmt

    dowhilestmt:	"do" stmt "while" "(" expr ")" ";"

    expr:		expr binop expr
            unop expr
            expr "[" expr "]"
            "(" expr ")"
            expr "(" exprs ")"
            NAME
            INT
            CHAR
            STRING

    exprs:		/*empty*/
            (expr ",")* expr

    binop:		"+" | "-" | "*" | "/" | "%" |
            "=" |
            "<" | "==" | "!="

    unop:		"!" | "-" | "*"

    type:		"int" stars
            "char" stars

    stars:		"*"*
